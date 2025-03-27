import os
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import miaplpy.io.AstronomicalHandBook as AstronomicalHandbook
import numbers
import math

# from mintpy.constants import EARTH_RADIUS, SPEED_OF_LIGHT
from mintpy.utils import isce_utils
from mintpy.objects import sensor

__all__ = ['extract_tops_metadata', 'extract_stripmap_metadata', 'extract_alosStack_metadata']


def extract_tops_metadata(xml_file):
    """
    Extract metadata from an XML file for Sentinel-1/TOPS.

    Parameters:
        xml_file (str): Path to the .xml file (e.g., 'reference/IW1.xml').

    Returns:
        dict: Metadata extracted from the XML file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Navigate through the XML structure and extract required fields
    meta = {}

    bursts = root.findall(".//component[@name='burst1']")
    if not bursts:
        raise ValueError("No bursts found in the XML file.")

    first_burst = bursts[0]
    last_burst = bursts[-1]

    meta['prf'] = float(first_burst.find(".//property[@name='pulserepetitionfrequency']/value").text)
    meta['startUTC'] = first_burst.find(".//property[@name='burststartutc']/value").text
    meta['stopUTC'] = last_burst.find(".//property[@name='burststoputc']/value").text
    meta['radarWavelength'] = float(first_burst.find(".//property[@name='radarwavelength']/value").text)
    meta['startingRange'] = float(first_burst.find(".//property[@name='startingrange']/value").text)
    meta['passDirection'] = first_burst.find(".//property[@name='passdirection']/value").text
    meta['polarization'] = first_burst.find(".//property[@name='polarization']/value").text
    meta['trackNumber'] = int(first_burst.find(".//property[@name='tracknumber']/value").text)
    meta['orbitNumber'] = int(first_burst.find(".//property[@name='orbitnumber']/value").text)
    meta['PLATFORM'] = 'sen'

    iw_str = 'IW2'
    if os.path.basename(xml_file).startswith('IW'):
        iw_str = os.path.splitext(os.path.basename(xml_file))[0]
    meta['azimuthResolution'] = sensor.SENSOR_DICT['sen'][iw_str]['azimuth_resolution']
    meta['rangeResolution']   = sensor.SENSOR_DICT['sen'][iw_str]['range_resolution']

    sensing_start_text = first_burst.find(".//property[@name='sensingstart']/value").text
    sensing_end_text = first_burst.find(".//property[@name='sensingstop']/value").text

    # Convert sensing time to seconds (example placeholder logic for CENTER_LINE_UTC)
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    sensing_start_dt = datetime.strptime(sensing_start_text, time_format)
    sensing_end_dt = datetime.strptime(sensing_end_text, time_format)
    meta['CENTER_LINE_UTC'] = (microsecond_time(sensing_start_dt) + microsecond_time(sensing_end_dt))/2

    meta['azimuthPixelSize'] = float(first_burst.find(".//property[@name='azimuthtimeinterval']/value").text)
    meta['rangePixelSize'] = float(first_burst.find(".//property[@name='rangepixelsize']/value").text)
    meta['beam_mode'] = 'IW'
    fburst = first_burst.find(".//property[@name='swathnumber']/value").text
    meta['swathNumber'] = int(fburst)

    # Extract state vectors
    state_vector_components = root.findall(".//component[@name='state_vectors']/component")
    
    # Interpolate position and velocity at sensing_mid_dt
    interpolated_positions, interpolated_velocities = interpolateHermite(state_vector_components,
                                                                      meta['CENTER_LINE_UTC'])

    meta['interpolatedPosition'] = interpolated_positions
    meta['interpolatedVelocity'] = interpolated_velocities

    Vs = np.linalg.norm(interpolated_velocities)   #satellite speed
    meta['satelliteSpeed'] = Vs
    meta['azimuthPixelSize'] = Vs * float(first_burst.find(".//property[@name='azimuthtimeinterval']/value").text)
    meta['rangePixelSize'] = float(first_burst.find(".//property[@name='rangepixelsize']/value").text)
    
    llh = xyz_to_llh(interpolated_positions)
    enumat = enubasis(llh)
    venu = np.dot(enumat.xyz_to_enu, interpolated_velocities)
    #Heading
    meta['HEADING'] = np.arctan2(venu[0,0], venu[0,1])
    meta['altitude'] = llh[2]
    meta['earthRadius'] = radiusOfCurvature(llh)

    return meta, f'burst_{fburst}'


def extract_stripmap_metadata(xml_file):
    return isce_utils.extract_stripmap_metadata(xml_file)

def extract_alosStack_metadata(xml_file):
    return isce_utils.extract_alosStack_metadata(xml_file)


def microsecond_time(sensing_time):
    out = sensing_time.hour * 3600 + \
    sensing_time.minute * 60 + \
    sensing_time.second + \
    sensing_time.microsecond / 1e6
    return out
    

def interpolateHermite(state_vector_components, sensing_time):
    
    times = []
    positions = []
    velocities = []
    time_format = "%Y-%m-%d %H:%M:%S"
    for sv in state_vector_components:
        position = sv.find(".//property[@name='position']/value").text
        velocity = sv.find(".//property[@name='velocity']/value").text
        time = sv.find(".//property[@name='time']/value").text

        pos = [float(coord) for coord in position.strip('[]').split(', ')]
        vel = [float(coord) for coord in velocity.strip('[]').split(', ')]
        times.append(microsecond_time(datetime.strptime(time, time_format)))
        positions.append(pos)
        velocities.append(vel)
    # Interpolate position and velocity at sensing_time
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    positions = np.array(positions)[sorted_indices]
    velocities = np.array(velocities)[sorted_indices]

    unique_times, unique_indices = np.unique(times, return_index=True)
    times = times[unique_indices]
    positions = positions[unique_indices]
    velocities = velocities[unique_indices]

    mid_time = sensing_time #.timestamp()
    interpolated_positions = []
    interpolated_velocities = []

    for i in range(3):  # Interpolate each coordinate (x, y, z)
        hermite = CubicHermiteSpline(times, positions[:, i], velocities[:, i])
        interpolated_positions.append(hermite(mid_time))
        interpolated_velocities.append(hermite.derivative()(mid_time))

    return interpolated_positions, interpolated_velocities


def xyz_to_llh(xyz):
    """xyz_to_llh(xyz): returns llh=(lat (deg), lon (deg), h (meters)) for the instance ellipsoid \n
    given the coordinates of a point at xyz=(z,y,z) (meters). \n
    Based on closed form solution of H. Vermeille, Journal of Geodesy (2002) 76:451-454. \n
    Handles simple list or tuples (xyz represents a single point) or a list of lists or tuples (xyz represents several points)"""

    a = AstronomicalHandbook.PlanetsData.ellipsoid['Earth']['WGS-84'].a
    e2 = AstronomicalHandbook.PlanetsData.ellipsoid['Earth']['WGS-84'].e2
    
    a2 = a**2
    e4 = e2**2
    # just to cast back to single list once done
    onePosition = False
    if isinstance(xyz[0],numbers.Real):
        xyz = [xyz]
        onePosition = True
    
    r_llh = [0]*3
    d_llh = [0]*3
    
    xy2 = xyz[0]**2+xyz[1]**2
    p = (xy2)/a2
    q = (1.-e2)*xyz[2]**2/a2
    r = (p+q-e4)/6.
    s = e4*p*q/(4.*r**3)
    t = (1.+s+math.sqrt(s*(2.+s)))**(1./3.)
    u = r*(1.+t+1./t)
    v = math.sqrt(u**2+e4*q)
    w = e2*(u+v-q)/(2.*v)
    k = math.sqrt(u+v+w**2)-w
    D = k*math.sqrt(xy2)/(k+e2)

    
    r_llh[0] = math.atan2(xyz[2],D)
    r_llh[1] = math.atan2(xyz[1],xyz[0])
    r_llh[2] = (k+e2-1.)*math.sqrt(D**2+xyz[2]**2)/k
    
    d_llh[0] = math.degrees(r_llh[0])
    d_llh[1] = math.degrees(r_llh[1])
    d_llh[2] = r_llh[2] 
    if onePosition:
        return d_llh[0]
    else:
        return d_llh


def enubasis(posLLH):
    """
    xyzenuMat = elp.enubasis(posLLH)
    Given an instance elp of an Ellipsoid LLH position (as a list) return the
    transformation matrices from the XYZ frame to the ENU frame and the
    inverse from the ENU frame to the XYZ frame.  
    The returned object is a namedtuple with numpy matrices in elements
    named 'enu_to_xyz' and 'xyz_to_enu'
    enu_to_xyzMat = (elp.enubasis(posLLH)).enu_to_xyz
    xyz_to_enuMat = (elp.enubasis(posLLH)).xyz_to_enu
    """

    r_lat = np.radians(posLLH[0])
    r_lon = np.radians(posLLH[1])

    r_clt = np.cos(r_lat)
    r_slt = np.sin(r_lat)
    r_clo = np.cos(r_lon)
    r_slo = np.sin(r_lon)

    r_enu_to_xyzMat = np.matrix([
        [-r_slo, -r_slt*r_clo, r_clt*r_clo],
        [ r_clo, -r_slt*r_slo, r_clt*r_slo],
        [ 0.0  ,  r_clt      , r_slt]])

    r_xyz_to_enuMat = r_enu_to_xyzMat.transpose()

    from collections import namedtuple
    enuxyzMat = namedtuple("enuxyzMat", "enu_to_xyz  xyz_to_enu")

    return enuxyzMat(r_enu_to_xyzMat, r_xyz_to_enuMat)


def radiusOfCurvature(llh, hdg=0):
    """
    radiusOfCurvature(llh,[hdg]): returns Radius of Curvature (meters)
    in the direction specified by hdg for the instance ellipsoid
    given a position llh=(lat (deg), lon (deg), h (meters)).
    If no heading is given the default is 0, or North.
    """

    a = AstronomicalHandbook.PlanetsData.ellipsoid['Earth']['WGS-84'].a
    e2 = AstronomicalHandbook.PlanetsData.ellipsoid['Earth']['WGS-84'].e2

    r_lat = math.radians(llh[0])
    r_hdg = math.radians(hdg)

    reast = eastRadiusOfCurvature(a, e2, llh)
    rnorth = northRadiusOfCurvature(a, e2, llh)

    #radius of curvature for point on ellipsoid
    rdir = (reast*rnorth)/(
        reast*math.cos(r_hdg)**2 + rnorth*math.sin(r_hdg)**2)

    #add height of the llh point
    return rdir + llh[2]

def eastRadiusOfCurvature(a, e2, llh):
    """eastRadiusOfCurvature(llh): returns Radius of Curvature (meters) \nin the East direction for the instance ellipsoid \ngiven a position llh=(lat (deg), lon (deg), h (meters))"""
    
    r_lat = math.radians(llh[0])

    reast = a/math.sqrt(1.0 - e2*math.sin(r_lat)**2)
    return reast

def northRadiusOfCurvature(a, e2, llh):
    """northRadiusOfCurvature(llh): returns Radius of Curvature (meters) \nin the North direction for the instance ellipsoid \ngiven a position llh=(lat (deg), lon (deg), h (meters))"""

    r_lat = math.radians(llh[0])

    rnorth = (a*(1.0 - e2))/(1.0 - e2*math.sin(r_lat)**2)**(1.5)
    return rnorth