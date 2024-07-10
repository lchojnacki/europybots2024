# SPDX-License-Identifier: BSD-3-Clause

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
    for angle_offset in (-45, -30, -15, 0, 15, 30, 45):
        angle = current_angle + angle_offset
        new_heading = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        new_speed_angle = np.degrees(np.arccos(np.dot(wind_heading, new_heading)))
        new_speed = np.abs(np.cos(np.radians(new_speed_angle / 2)))
        vectors.append(compute_ship_speed_vector(new_heading, new_speed))


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
            if dist < 2.0 * ch.radius + jump:
                instructions.sail = min(ch.radius / jump, 1)
            else:
                instructions.sail = 1.0
            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
            if not ch.reached:
                instructions.location = Location(
                    longitude=ch.longitude, latitude=ch.latitude
                )
                break

        return instructions
