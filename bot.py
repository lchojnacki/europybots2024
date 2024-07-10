# SPDX-License-Identifier: BSD-3-Clause
import random
# flake8: noqa F401
from collections.abc import Callable

import numpy as np

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface


def wind_direction(u: float, v: float) -> float:
    """
    Calculate the meteorological wind direction based on the horizontal wind
    components.

    Args:
        u: Horizontal wind component in the x-direction.
        v: Horizontal wind component in the y-direction.

    Returns:
        The meteorological wind direction in degrees.
    """

    # Calculate the angle using atan2
    theta = np.atan2(v, u)

    # Convert the angle from radians to degrees
    theta_degrees = np.degrees(theta)

    # Convert to meteorological wind direction
    direction = (270 - theta_degrees) % 360

    return direction


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the absolute difference between two angles and normalize it to be
    within 0 to 360 degrees.

    Args:
        angle1: The first angle in degrees.
        angle2: The second angle in degrees.

    Returns:
        The normalized absolute difference between the two angles in degrees.
    """

    # Calculate the absolute difference using numpy
    diff = np.abs(angle1 - angle2)

    # Normalize the difference to be within 0 to 360 degrees
    diff = diff % 360

    # Adjust if the difference is greater than 180 degrees
    diff: float = np.where(diff > 180, 360 - diff, diff)

    return diff


def should_change_direction(wind_angle, ship_angle, max_angle: int = 100) -> bool:
    """
    Determine if the ship should change its direction based on the difference
    between the wind angle and the ship's angle.

    Args:
        wind_angle: The angle of the wind in degrees.
        ship_angle: The ship's current angle in degrees.
        max_angle: The maximum allowable angle difference for not changing
                   direction (default is 100 degrees).

    Returns:
        A boolean indicating whether the ship should change its direction.
    """

    return angle_difference(wind_angle, ship_angle) > max_angle

def compute_ship_speed_vector(heading: np.ndarray, speed: float) -> np.ndarray:
    """
    Compute the speed vector from the heading and speed.

    Parameters
    ----------
    heading:
        The heading of the ship.
    speed:
        The speed of the ship.

    Returns
    -------
    vector:
        The speed vector.
    """
    return speed * heading


def compute_speed_vectors_for_angles(
    ship_heading: np.ndarray, wind_heading: np.ndarray
) -> list[np.ndarray]:
    """
    Compute the speed vectors for the ship at different angles.

    Parameters
    ----------
    ship_heading:
        The heading of the ship.
    wind_heading:
        The heading of the wind.

    Returns
    -------
    vectors:
        The speed vectors for the ship at different angles.
    """
    vectors = []
    current_angle = np.degrees(np.arccos(np.dot(ship_heading, wind_heading)))
    for angle_offset in (-30, -15, 0, 15, 30):
        angle = current_angle + angle_offset
        new_heading = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        new_speed_angle = np.degrees(np.arccos(np.dot(wind_heading, new_heading)))
        new_speed = np.abs(np.cos(np.radians(new_speed_angle / 2)))
        vectors.append(compute_ship_speed_vector(new_heading, new_speed))
    return vectors


def compute_proposed_new_ship_locations(
    location: Location,
    ship_heading: np.ndarray,
    wind_heading: np.ndarray,
    dt: float,
) -> list[Location]:
    """
    Compute the proposed new ship positions for different angles.

    Parameters
    ----------
    location:
        The current location of the ship.
    ship_heading:
        The heading of the ship.
    wind_heading:
        The heading of the wind.
    dt:
        The time step in hours.

    Returns
    -------
    locations:
        The proposed new ship positions for different angles.
    """
    locations = []
    for v in compute_speed_vectors_for_angles(ship_heading, wind_heading):
        new_location_vec = np.asarray([location.longitude, location.latitude]) + v*dt
        new_longitude, new_latitude = new_location_vec
        locations.append(Location(longitude=new_longitude, latitude=new_latitude))
    return locations


def compute_best_ship_angle(
    location: Location,
    ship_heading: np.ndarray,
    wind_heading: np.ndarray,
    destination: Location,
    dt: float,
) -> float:
    """
    Compute the best ship position to reach the destination.

    We first compute the proposed new ship positions for different angles.
    Then at each proposed location, we compute the heading to the destination.
    For each heading, we compute the speed value based on the wind direction.
    Finally, we select the location that will take us to the destination the fastest.

    Parameters
    ----------
    location:
        The current location of the ship.
    ship_heading:
        The heading of the ship.
    wind_heading:
        The heading of the wind.
    destination:
        The destination location.
    dt:
        The time step in hours.

    Returns
    -------
    best_location:
        The best ship position to reach the destination.
    """
    wind_heading = wind_heading / np.linalg.norm(wind_heading)
    proposed_locations = compute_proposed_new_ship_locations(
        location=location,
        ship_heading=ship_heading,
        wind_heading=wind_heading,
        dt=dt,
    )
    best_angle = None
    best_time = np.inf
    for proposed_location, angle in zip(proposed_locations, (-30, -15, 0, 15, 30)):
        heading = np.array(
            [
                destination.longitude - proposed_location.longitude,
                destination.latitude - proposed_location.latitude,
            ]
        )
        heading = heading / np.linalg.norm(heading)
        speed = np.abs(np.cos(np.radians(np.degrees(np.arccos(np.dot(wind_heading, heading))) / 2)))
        dist = distance_on_surface(
            longitude1=proposed_location.longitude,
            latitude1=proposed_location.latitude,
            longitude2=destination.longitude,
            latitude2=destination.latitude,
        )
        time = dist / speed if speed > 0 else np.inf
        if time < best_time:
            best_time = time
            best_angle = angle
    return best_angle


class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "The Ifers"  # This is your team name
        # This is the course that the ship has to follow
        self.course = [
            Checkpoint(latitude=43.797109, longitude=-11.264905, radius=50),
            Checkpoint(longitude=-29.908577, latitude=17.999811, radius=50),
            Checkpoint(latitude=-11.441808, longitude=-29.660252, radius=50),
            Checkpoint(longitude=-63.240264, latitude=-61.025125, radius=50),
            Checkpoint(latitude=2.806318, longitude=-168.943864, radius=1990.0),
            Checkpoint(latitude=-62.052286, longitude=169.214572, radius=50.0),
            Checkpoint(latitude=-15.668984, longitude=77.674694, radius=1190.0),
            Checkpoint(latitude=-39.438937, longitude=19.836265, radius=50.0),
            Checkpoint(latitude=14.881699, longitude=-21.024326, radius=50.0),
            Checkpoint(latitude=44.076538, longitude=-18.292936, radius=50.0),
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            ),
        ]

    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()

        # TODO: Remove this, it's only for testing =================
        current_wind = forecast(latitudes=latitude, longitudes=longitude, times=0)
        wind_angle = wind_direction(*current_wind)
        ship_angle = wind_direction(*vector)

        current_position_terrain = world_map(latitudes=latitude, longitudes=longitude)
        # ===========================================================

        # Go through all checkpoints and find the next one to reach
        for ch in self.course:
            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            # Consider slowing down if the checkpoint is close
            jump = dt * np.linalg.norm(speed)
            if should_change_direction(wind_angle, ship_angle):
                # TODO: Turn left or right based on the wind direction
                instructions.left = 10.0
                break
            if dist < 2.0 * ch.radius + jump:
                instructions.sail = min(ch.radius / jump, 1)
            else:
                instructions.sail = 1.0
            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
            if not ch.reached:
                # instructions.location = Location(
                #     longitude=ch.longitude, latitude=ch.latitude
                # )
                angle = compute_best_ship_angle(
                    location=Location(longitude=longitude, latitude=latitude),
                    ship_heading=np.asarray(vector),
                    wind_heading=np.asarray(current_wind),
                    destination=Location(longitude=ch.longitude, latitude=ch.latitude),
                    dt=dt,
                )
                if angle is not None and random.random() < 0.5:
                    if angle < 0:
                        instructions.left = abs(angle)
                    else:
                        instructions.right = abs(angle)
                else:
                    instructions.location = Location(
                        longitude=ch.longitude, latitude=ch.latitude
                    )
                break
        return instructions
