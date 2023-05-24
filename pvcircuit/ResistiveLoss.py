"""
Class to calculate resistive losses

Example:
    See main file and notebooks for examples.

Todo:

Dev infos:
.. use Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

"""

from __future__ import annotations

import logging
import pprint

import matplotlib.pyplot as plt
import numpy as np
from dataeval.iv_factory import IVCollection
from dataeval.qe_factory import EQECollection
from pvlib.ivtools.sde import fit_sandia_simple
from shapely.geometry import Polygon

import pvcircuit as pvc

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

FORMAT = "%(asctime)s %(clientip)-15s %(user)-8s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MonolithicalInterconnection:
    """
    class to calculate the optical and series resistance losses for monolithical series interconnection of the cells
    """

    def __init__(
        self,
        front_layer_sheet_resistance: float = 100.0,  # [Ohm / sq] Resistivity of the TCO/conductor front layer e.g. 4.44e-4 [Ohm cm] from Sans IZO batch
        rear_layer_sheet_resistance: float = 100.0,  # [Ohm cm] Resistivity of the TCO/conductor rear layer
        L: float = 10,  # [cm] length of the cell
        wd: float = 530e-4,  # [cm] dead area width
        wa: float = 1.0,  # [cm] active area width
    ):
        self.front_layer_sheet_resistance = front_layer_sheet_resistance
        self.rear_layer_sheet_resistance = rear_layer_sheet_resistance
        self.L = L
        self.wd = wd
        self.wa = wa

    def __str__(self):
        return f"""
        Cell:
            L = {self.L} [cm]
            wa = {self.wa} [cm]
            wd = {self.wd} [cm]
            cell_area = {self.cell_area} [cm^2]
        Front side:
            front_layer_sheet_resistance = {self.front_layer_sheet_resistance} [Ohm / sq]
        Rear side:
            rear_layer_sheet_resistance = {self.rear_layer_sheet_resistance} [Ohm / sq]
        """

    def __repr__(self):
        return self.__str__()

    @property
    def cell_area(self) -> float:
        """
        calculate the cell area
        """
        return (self.wa + self.wd) * self.L

    def monolith_series_resistance(self) -> float:
        """
        calculate series resistance contribution from monolithical series interconnection
        returns rs:float = series reistance [Ohm cm^2]
        """
        rs_front = 1 / 3 * self.front_layer_sheet_resistance * self.wa**3 / (self.wa + self.wd)
        rs_rear = 1 / 3 * self.rear_layer_sheet_resistance * self.wa**3 / (self.wa + self.wd)
        rs = rs_front + rs_rear
        return rs

    def monolith_series_powerloss(self, jmpp: float, vmpp: float) -> float:
        """
        calculate series resistance contribution from monolithical series interconnection
        returns rs:float = series reistance [Ohm cm^2]
        """
        rs_front = 1 / 3 * jmpp / vmpp * self.front_layer_sheet_resistance * self.wa**3 / (self.wa + self.wd)
        rs_rear = 1 / 3 * jmpp / vmpp * self.rear_layer_sheet_resistance * self.wa**3 / (self.wa + self.wd)
        fr = rs_front + rs_rear

        fa = self.wa / (self.wa + self.wd)

        return fr + fa

    def monolith_current_loss(self, jsc: float) -> float:
        """
        Calculates the current density of the full cell area form active and dead area

        Args:
            jsc (float): Short circuit current density of the active area

        Returns:
            float: Short circuit current density of the full cell area
        """
        # wa = np.linspace(0.01, 2.5)  # cm
        # R_tco = 7.2  # Ohm/sq
        dead_area_loss = self.wa / (self.wa + self.wd)

        return dead_area_loss * jsc

    def monolith_loss_factor(self, pmpp, jmpp, vmpp):
        """
        Calculate the power loss base on loss factor simulation.
        """
        f_loss = self.wa / (self.wa + self.wd) - 1 / 3 * jmpp / vmpp * self.front_layer_sheet_resistance * self.wa**3 / (
            self.wa + self.wd
        )
        return (1 - f_loss) * pmpp


class GridInterconnection:
    """
    Class to simulate the optical and series resistance losses of a cell with front and rear side grid.
    """

    def __init__(
        self,
        front_layer_sheet_resistance: float = 100,  # [Ohm/sq] Resistivity of the TCO/conductor front layer
        rear_layer_sheet_resistance: float = 100,  # [Ohm/sq] Resistivity of the TCO/conductor rear layer
        front_metal_finger_number=4,  # Number of front metal fingers
        front_metal_finger_rho: float = 1.5e-6,  # [Ohm cm] Resistivity of the front metal contact, e.g. Silver (Ag) 1.59e-6 Ohm cm
        front_metal_finger_y=30e-4,  # [cm] front metal dimension in y-direction aka front metal finger width.
        front_metal_finger_thickness: float = 5e-4,  # [cm] front metal thickness
        rear_metal_finger_number=4,  # Number of rear metal fingers
        rear_metal_finger_rho: float = 1.5e-6,  # [Ohm cm] Resistivity of the rear metal contact, e.g. Silver (Ag) 1.59e-6 Ohm cm
        rear_metal_finger_y=30e-4,  # [cm] rear metal dimension in y-direction aka rear metal finger width.
        rear_metal_finger_thickness: float = 5e-4,  # [cm] rear metal thickness
        # cell dimensions
        cell_x: float = 0.5,  # [cm] cell dimension in x-direction
        cell_y: float = 0.5,  # [cm] cell dimension in y-direction
    ):
        # Resistivities
        self.front_layer_sheet_resistance = front_layer_sheet_resistance
        self.rear_layer_sheet_resistance = rear_layer_sheet_resistance

        self.front_metal_finger_rho = front_metal_finger_rho
        self.rear_metal_finger_rho = rear_metal_finger_rho

        # Cell dimensions
        self.cell_x = cell_x  # [cm]
        self.cell_y = cell_y  # [cm]

        # front metal geometry
        self.front_metal_finger_number = front_metal_finger_number
        self.front_metal_finger_x = self.front_metal_grid_x  # Finger extend over full grid
        self.front_metal_finger_y = front_metal_finger_y
        self.front_metal_finger_thickness = front_metal_finger_thickness

        self.front_busbar_number = 1
        self.front_busbar_x = 0.0e-1  # [cm]

        # rear metal geometry
        self.rear_metal_finger_number = rear_metal_finger_number
        self.rear_metal_finger_x = self.rear_metal_grid_x  # Finger extend over full grid
        self.rear_metal_finger_y = rear_metal_finger_y
        self.rear_metal_finger_thickness = rear_metal_finger_thickness

        self.rear_busbar_number = 1
        self.rear_busbar_x = 0

    # front unit cell dimension

    # def __repr__(self):
    # pp = pprint.PrettyPrinter(indent=4)
    # attrs = {}
    # for attr in dir(self):
    #     if not attr.startswith('__'):
    #         try:
    #             attrs[attr] = getattr(self, attr)
    #         except Exception as e:
    #             attrs[attr] = f'<cannot print: {e}>'
    # return pp.pformat(attrs)

    # pp = pprint.PrettyPrinter(indent=4)
    # class_attrs = dir(self)
    # class_vals = [getattr(self,attr) for attr in class_attrs if not attr.startswith('__')]
    # print(class_vals)
    # return pp.pformat([f"{attr}" for attr in class_attrs if not attr.startswith('__')])

    def __str__(self):
        return f"""
        Cell:
            cell_x = {self.cell_x} [cm]
            cell_y = {self.cell_y} [cm]
            cell_area = {self.cell_area} [cm^2]
        Front side:
            front_layer_sheet_resistance = {self.front_layer_sheet_resistance} [Ohm / sq]
            front_metal_finger_number = {self.front_metal_finger_number} [#]
            front_metal_finger_rho = {self.front_metal_finger_rho} [Ohm cm]
            front_metal_finger_y = {self.front_metal_finger_y} [cm]
            front_metal_finger_thickness = {self.front_metal_finger_thickness} [cm]
        Rear side:
            rear_layer_sheet_resistance = {self.rear_layer_sheet_resistance} [Ohm / sq]
            rear_metal_finger_number = {self.rear_metal_finger_number} [#]
            rear_metal_finger_rho = {self.rear_metal_finger_rho} [Ohm cm]
            rear_metal_finger_y = {self.rear_metal_finger_y} [cm]
            rear_metal_finger_thickness = {self.rear_metal_finger_thickness} [cm]
        """

    def __repr__(self):
        return self.__str__()

    @property
    def front_metal_grid_x(self):
        return self.cell_x - 2 * 0.1e-1  # [cm]

    @property
    def front_metal_grid_y(self):
        return self.cell_y  # [cm]

    @property
    def rear_metal_grid_x(self):
        return self.cell_x - 2 * 0.1e-1  # [cm]

    @property
    def rear_metal_grid_y(self):
        return self.cell_y  # [cm]

    @property
    def front_unit_cell_x(self):
        """
        Size of the front unit cell in x direction, which is the finger length including half the width of the busbar.
        """
        # return 0.5 * self.cell_x / self.front_busbar_number
        return 0.5 * self.front_metal_grid_x / self.front_busbar_number

    @property
    def front_unit_cell_y(self):
        """
        Size of the front unit cell in y direction, which is equivalent to the finger pitch. It's the full pitch to not split the finger width.
        """
        return self.front_metal_grid_y / self.front_metal_finger_number

    @property
    def front_unit_cell_metal_finger_x(self):
        """
        Length of the unit cell finger, which is half the distance between busbars, without the busbar
        """
        return 0.5 * self.front_metal_finger_x / self.front_busbar_number - 0.5 * self.front_busbar_x

    @property
    def front_metal_finger_pitch(self):
        """reutuns the fron metal finger pitch"""
        return self.front_metal_grid_y / self.front_metal_finger_number

    @property
    def front_busbar_pitch(self):
        """reutuns the fron metal finger pitch"""
        return self.front_metal_finger_x / self.front_busbar_number

    # rear unit cell dimension

    @property
    def rear_unit_cell_x(self):
        """
        Size of the rear unit cell in x direction, which is the finger length including half the width of the busbar.
        """
        # return 0.5 * self.cell_x / self.rear_busbar_number
        return 0.5 * self.rear_metal_grid_x / self.rear_busbar_number

    @property
    def rear_unit_cell_y(self):
        """
        Size of the rear unit cell in y direction, which is equivalent to the finger pitch. It's the full pitch to not split the finger width.
        """
        return self.rear_metal_grid_y / self.rear_metal_finger_number

    @property
    def rear_unit_cell_metal_x(self):
        """
        Length of the unit cell finger, which is half the distance between busbars
        """
        return 0.5 * self.rear_metal_grid_x / self.rear_busbar_number - 0.5 * self.rear_busbar_x

    @property
    def rear_metal_finger_pitch(self):
        """Returns the rear metal finger pitch"""
        return self.rear_metal_grid_y / self.rear_metal_finger_number

    @property
    def cell_area(self):
        """Calculate the cell area"""
        return self.cell_x * self.cell_y  # [cm^2]

    def top_grid_current_loss(self, jsc: float) -> float:
        """
        Calculates the current density considering shading of the cell's front side by the metal grid

        Args:
            jsc (float): Short circuit current density of the active area [mA/cm^2]

        Returns:
            float: Short circuit current density of the full cell area [mA/cm^2]
        """
        finger_area = self.front_metal_finger_number * self.front_metal_finger_x * self.front_metal_finger_y
        metal_fraction = finger_area / self.cell_area
        logger.debug("Metal fraction = %.2f", metal_fraction * 100)
        return (1 - metal_fraction) * jsc

    def bottom_grid_current_loss(self, jsc: float) -> float:
        """
        Calculates the current density considering shading of the cell's front and rear side by the metal grid

        Args:
            jsc (float): Short circuit current density of the active area [mA/cm^2]

        Returns:
            float: Short circuit current density of the full cell area [mA/cm^2]
        """
        front_finger_area = self.front_metal_finger_number * self.front_metal_finger_x * self.front_metal_finger_y
        front_metal_fraction = front_finger_area / self.cell_area
        logger.debug("Metal fraction = %.2f", front_metal_fraction * 100)

        rear_finger_area = self.rear_metal_finger_number * self.rear_metal_finger_x * self.rear_metal_finger_y
        rear_metal_fraction = rear_finger_area / self.cell_area
        logger.debug("Rera metal fraction = %.2f", rear_metal_fraction * 100)

        if ((self.front_metal_finger_number / self.rear_metal_finger_number) % 2 == 1) or ((
            self.rear_metal_finger_number / self.front_metal_finger_number
        ) % 2 == 1):
            # finger cover
            metal_fraction = max(front_metal_fraction, rear_metal_fraction)
        else:
            # finger don't cover
            metal_fraction = front_metal_fraction + rear_metal_fraction

        return (1 - metal_fraction) * jsc

    def grid_series_resistance(self):
        """
        calculate series resistance contribution from metal grid.
        Area for this domain is half the finger length and finger pitch
        returns rs:float = series resistance [Ohm cm^2]
        """

        # Front metal finger line resistivity
        r_line_front_metal = (
            self.front_metal_finger_rho / self.front_metal_finger_y / self.front_metal_finger_thickness
        )  # [Ohm / cm]
        logger.debug(
            "Front metal finger sheet resistance = %.2f mOhm / sq",
            self.front_metal_finger_rho / self.front_metal_finger_thickness * 1e3,
        )

        # contribution of the front metal finger
        rs_front_metal = (
            1 / 3 * r_line_front_metal * self.front_unit_cell_metal_finger_x * self.front_unit_cell_x * self.front_unit_cell_y
        )  # [Ohm cm^2]

        # contribution of the front layer sheet resistance
        logger.debug("Front layer sheet resistance = %.2f mOhm / sq", self.front_layer_sheet_resistance)

        rs_front_layer = (
            1
            / 12
            * self.front_layer_sheet_resistance
            * self.front_unit_cell_x
            * self.front_unit_cell_y
            * (self.front_unit_cell_y - self.front_metal_finger_y)
            / self.front_unit_cell_metal_finger_x
        )  # [Ohm cm^2]

        # contribution of the contact between front layer and metal finger
        rc_front_metal2layer = 0  # [Ohm cm^2]

        # Rear metal finger line resistivity
        r_line_rear_metal = self.rear_metal_finger_rho / self.rear_metal_finger_y / self.rear_metal_finger_thickness  # [Ohm / cm]
        logger.debug(
            "Rear metal finger sheet resistance = %.2f mOhm / sq",
            self.rear_metal_finger_rho / self.rear_metal_finger_thickness * 1e3,
        )

        # contribution of the rear metal finger
        rs_rear_metal = (
            1 / 3 * r_line_rear_metal * self.rear_unit_cell_metal_x * self.rear_unit_cell_x * self.rear_unit_cell_y
        )  # [Ohm cm^2]

        # contribution of the rear layer sheet resistance
        logger.debug("Rear layer sheet resistance = %.2f mOhm / sq", self.rear_layer_sheet_resistance)

        rs_rear_layer = (
            1
            / 12
            * self.rear_layer_sheet_resistance
            * self.rear_unit_cell_x
            * self.rear_unit_cell_y
            * (self.rear_unit_cell_y - self.rear_metal_finger_y)
            / self.rear_unit_cell_metal_x
        )  # [Ohm cm^2]

        # contribution of the contact between rear layer and metal finger
        rc_rear_metal2layer = 0  # [Ohm cm^2]

        rs_front = rs_front_metal + rs_front_layer + rc_front_metal2layer  # [Ohm cm^2]
        rs_rear = rs_rear_metal + rs_rear_layer + rc_rear_metal2layer  # [Ohm cm^2]
        # rs_rear = 0  # [Ohm cm^2]

        logger.debug("Front metal resistance = %.2f Ohm cm^2", rs_front)
        logger.debug("Rear metal resistance = %.2f Ohm cm^2", rs_rear)

        rs_total = rs_front + rs_rear  # [Ohm cm^2]
        logger.debug("Total grid resistance = %.2f Ohm cm^2", rs_total)

        return rs_total

    def draw_grid(self):
        """
        Draw the grid
        """
        _, ax = plt.subplots(1)

        cell = Polygon(
            [
                (self.cell_x * 0.5, self.cell_y * 0.5),
                (-self.cell_x * 0.5, self.cell_y * 0.5),
                (-self.cell_x * 0.5, -self.cell_y * 0.5),
                (self.cell_x * 0.5, -self.cell_y * 0.5),
            ]
        )

        front_metal_grid = Polygon(
            [
                (self.front_metal_grid_x * 0.5, self.front_metal_grid_y * 0.5),
                (-self.front_metal_grid_x * 0.5, self.front_metal_grid_y * 0.5),
                (-self.front_metal_grid_x * 0.5, -self.front_metal_grid_y * 0.5),
                (self.front_metal_grid_x * 0.5, -self.front_metal_grid_y * 0.5),
            ]
        )

        rear_metal_grid = Polygon(
            [
                (self.rear_metal_grid_x * 0.5, self.rear_metal_grid_y * 0.5),
                (-self.rear_metal_grid_x * 0.5, self.rear_metal_grid_y * 0.5),
                (-self.rear_metal_grid_x * 0.5, -self.rear_metal_grid_y * 0.5),
                (self.rear_metal_grid_x * 0.5, -self.rear_metal_grid_y * 0.5),
            ]
        )

        # ax.fill(*cell.exterior.xy, fc="None", ec="b")
        ax.fill(*front_metal_grid.exterior.xy, fc="None", ec="b")
        ax.fill(*rear_metal_grid.exterior.xy, fc="None", ec="r")

        dx = self.rear_metal_finger_x * 0.5
        dy = self.rear_metal_finger_y * 0.5
        center_point_x = 0
        for finger_nr in range(self.rear_metal_finger_number):
            center_point_y = (
                -0.5 * self.rear_metal_grid_y + 0.5 * self.rear_metal_finger_pitch + finger_nr * self.rear_metal_finger_pitch
            )
            finger = Polygon(
                [
                    (center_point_x + dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y - dy),
                    (center_point_x + dx, center_point_y - dy),
                ]
            )
            ax.fill(*finger.exterior.xy, fc=(1, 0, 0, 1), ec=None)

        dx = self.front_metal_finger_x * 0.5
        dy = self.front_metal_finger_y * 0.5
        center_point_x = 0

        for finger_nr in range(self.front_metal_finger_number):
            center_point_y = (
                -0.5 * self.front_metal_grid_y + 0.5 * self.front_metal_finger_pitch + finger_nr * self.front_metal_finger_pitch
            )
            finger = Polygon(
                [
                    (center_point_x + dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y - dy),
                    (center_point_x + dx, center_point_y - dy),
                ]
            )
            ax.fill(*finger.exterior.xy, fc=(0, 0, 1, 1), ec=None)

        dx = 0.5 * self.front_busbar_x
        dy = 0.5 * self.front_metal_grid_y
        center_point_y = 0

        for busbar_nr in range(self.front_busbar_number):
            center_point_x = -0.5 * self.front_metal_grid_x + 0.5 * self.front_busbar_pitch + busbar_nr * self.front_busbar_pitch
            busbar = Polygon(
                [
                    (center_point_x + dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y - dy),
                    (center_point_x + dx, center_point_y - dy),
                ]
            )
            ax.fill(*busbar.exterior.xy, fc=(0, 0, 1, 0.5), ec="None")

        # Draw unit cell
        if self.front_metal_finger_number > 0:
            dx = self.front_unit_cell_x * 0.5
            dy = self.front_unit_cell_y * 0.5

            center_point_x = (
                -0.5 * self.front_metal_grid_x
                + 0.25 * self.front_busbar_pitch
                + np.floor(0.5 * self.front_busbar_number) * self.front_busbar_pitch
            )
            center_point_y = (
                -0.5 * self.front_metal_grid_y
                + 0.5 * self.front_metal_finger_pitch
                + np.floor(0.5 * self.front_metal_finger_number) * self.front_metal_finger_pitch
            )

            finger = Polygon(
                [
                    (center_point_x + dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y + dy),
                    (center_point_x - dx, center_point_y - dy),
                    (center_point_x + dx, center_point_y - dy),
                ]
            )
            ax.fill(*finger.exterior.xy, fc=(1, 0, 0, 0.3), ec="r")

        ax.set_box_aspect(1)


class Tandem4T:
    """
    PVCircuit Tandem class. Paramters are here extracted from the provided IV and EQE files.
    #TODO Implement functionality into PVCircuit
    """

    def __init__(self, nr_of_junctions) -> None:
        # A = best_iv.area  # [cm^2]
        self.TC = 25  # [degC]
        # Eg = qe_collection.bandgaps["Rau_bandgap_eV"].mean()  # [eV]
        self.TC_REF = 25  # [degC]

        self.tandem = pvc.Multi2T(name="Psk", Rs2T=1, Eg_list=[1] * nr_of_junctions, n=[1], J0ratio=[1], Jext=1, area=1)

        self.iv = [0] * nr_of_junctions

    def load_iv_folder(self, folder, junction_nr):
        """
        Loads all IV from a folder and uses the IV with highest eta from the set.
        """
        cell_collection = IVCollection.from_folder(folder)
        cell_collection.make_mean_from_fwd_rev()

        best_pixel = cell_collection.get_etas()
        best_pixel = best_pixel[best_pixel["sweep_direction"] == "MEAN"]
        best_pixel = best_pixel.loc[best_pixel["etas"].idxmax()]

        best_iv = cell_collection[best_pixel.name]

        psc_params = fit_sandia_simple(  # [A/cm^2]  # [A/cm^2]  # [Ohm cm^2]  # [Ohm cm^2]
            best_iv.voltage.values, best_iv.current.values
        )
        ilim = 0.6
        while psc_params[2] < 0:
            ilim -= 0.01
            # fit iv
            psc_params = fit_sandia_simple(  # [A/cm^2]  # [A/cm^2]  # [Ohm cm^2]  # [Ohm cm^2]
                best_iv.voltage.values, best_iv.current.values, ilim=ilim
            )

        (isc, io, rs, rsh, nNsVth) = fit_sandia_simple(best_iv.voltage.values, best_iv.current.values, ilim=ilim)

        jo_scale = 1000
        jsc = isc / best_iv.measurement_area  # [A/cm^2]
        Rser = rs * best_iv.measurement_area
        Gsh = 1 / (rsh * best_iv.measurement_area)
        jo = io / best_iv.measurement_area
        n = nNsVth / pvc.junction.Vth(self.TC)
        jdb = pvc.junction.Jdb(TC=self.TC, Eg=self.tandem.j[junction_nr].Eg)
        joratio = jo_scale * jo / (jdb * jo_scale) ** (1.0 / n)

        self.tandem.j[junction_nr].set(Jext=jsc, n=[n], J0ratio=[joratio], Rser=Rser, Gsh=Gsh, area=best_iv.measurement_area)
        self.tandem.Rs2T = Rser
        self.iv[junction_nr] = best_iv

    def load_iv_files(self, filepaths, junction_nr):
        """
        Loads IV form parameters from given filepaths.
        """
        # check if filepaths is a list
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        cell_collection = IVCollection.from_filepaths(filepaths)
        cell_collection.make_mean_from_fwd_rev()

        best_etas = cell_collection.get_etas().reset_index().groupby("sweep_direction").max()
        if best_etas.index.str.contains("MEAN")[0]:
            best_cell = best_etas.loc["MEAN"]
        else:
            best_cell = best_etas.iloc[0]
        best_iv = cell_collection[best_cell["index"]]
        # [A]  # [A]  # [Ohm]  # [Ohm]
        (isc, io, rs, rsh, nNsVth) = fit_sandia_simple(best_iv.voltage.values, best_iv.current.values)

        jo_scale = 1000
        jsc = isc / best_iv.measurement_area  # [A/cm^2]
        Rser = rs * best_iv.measurement_area
        Gsh = 1 / (rsh * best_iv.measurement_area)
        jo = io / best_iv.measurement_area
        n = nNsVth / pvc.junction.Vth(self.TC)
        jdb = pvc.junction.Jdb(TC=self.TC, Eg=self.tandem.j[junction_nr].Eg)
        joratio = jo_scale * jo / (jdb * jo_scale) ** (1.0 / n)
        # TODO get eg from EQE
        self.tandem.j[junction_nr].set(Eg=1.12, Jext=jsc, n=[n], J0ratio=[joratio], Rser=Rser, Gsh=Gsh)

    def load_eqe_files(self, filepaths, juntion_nr):
        """
        Loads the eq parameters from the given filepath
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        qe_collection = EQECollection.from_filepaths(filepaths)
        qe_collection.get_bandgaps()
        eg = qe_collection.bandgaps["Rau_bandgap_eV"].mean()
        self.tandem.j[juntion_nr].set(Eg=eg)

    def load_eqe_folder(self, folder, juntion_nr):
        """
        Loads the eq parameters from the given filepath
        """
        qe_collection = EQECollection.from_folder(folder)
        qe_collection.get_bandgaps()
        self.tandem.j[juntion_nr].set(Eg=qe_collection.bandgaps["Rau_bandgap_eV"].mean())
        qe_collection = EQECollection.from_folder(folder)
        qe_collection.get_bandgaps()
        self.tandem.j[juntion_nr].set(Eg=qe_collection.bandgaps["Rau_bandgap_eV"].mean())
