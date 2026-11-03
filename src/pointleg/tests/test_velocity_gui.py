"""Check velocity GUI bounds using handles without a server or physics env."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from mjlab.tasks.registry import load_env_cfg
from test_audit_fixes import command_env


class GuiHandle:
    def __init__(self, initial_value=None, **kwargs):
        self.value = initial_value
        self.__dict__.update(kwargs)
        self.callback = None

    def on_update(self, callback):
        self.callback = callback
        return callback

    on_click = on_update


class Gui:
    def __init__(self):
        self.inputs = {}

    def add_folder(self, name):
        return nullcontext()

    def add_slider(self, label, *, initial_value, min, max, step):
        if not min <= initial_value <= max or not 0 < step <= max - min:
            raise AssertionError(f"Invalid slider: {label}")
        handle = GuiHandle(initial_value, min=min, max=max, step=step)
        self.inputs[label] = handle
        return handle

    def add_number(self, label, **kwargs):
        handle = GuiHandle(**kwargs)
        self.inputs[label] = handle
        return handle

    def add_checkbox(self, label, **kwargs):
        return self.add_number(label, **kwargs)

    def add_button(self, label, **kwargs):
        return self.add_number(label, **kwargs)


def panel(x_range=(.15, .25), y_range=(0., 0.), yaw_range=(0., 0.)):
    cfg = load_env_cfg("PointLeg-HighStep-v0", play=True)
    cfg.commands["twist"].ranges.lin_vel_x = x_range
    cfg.commands["twist"].ranges.lin_vel_y = y_range
    cfg.commands["twist"].ranges.ang_vel_z = yaw_range
    command = cfg.commands["twist"].build(command_env(cfg, count=4))
    gui = Gui()
    command.create_gui("twist", SimpleNamespace(gui=gui), lambda: 0)
    return command, gui.inputs


class VelocityGuiTests(unittest.TestCase):
    def test_high_step_zero_axes_are_disabled_and_forward_axis_cannot_reverse(self):
        command, inputs = panel()
        self.assertEqual(len(command._joystick_sliders), 3)
        self.assertEqual(inputs["lin_vel_x"].min, 0.)
        self.assertEqual(inputs["lin_vel_x"].max, .25)
        for name in ("lin_vel_y", "ang_vel_z"):
            self.assertTrue(inputs[name].disabled)
            self.assertEqual(inputs[name].value, 0.)
            self.assertNotIn(f"Max {name}", inputs)

    def test_narrow_and_negative_only_ranges_have_valid_slider_bounds(self):
        _, inputs = panel(x_range=(-.02, -.01))
        self.assertEqual(inputs["lin_vel_x"].min, -.02)
        self.assertEqual(inputs["lin_vel_x"].max, 0.)
        self.assertEqual(inputs["Max lin_vel_x"].value, .02)
        inputs["Max lin_vel_x"].value = .01
        inputs["lin_vel_x"].value = -.02
        inputs["Max lin_vel_x"].callback(None)
        self.assertEqual(inputs["lin_vel_x"].min, -.01)
        self.assertEqual(inputs["lin_vel_x"].value, -.01)

    def test_bidirectional_axis_limits_and_callbacks_remain_independent(self):
        _, inputs = panel(x_range=(-.4, .6), y_range=(-.3, .3), yaw_range=(-.5, .5))
        self.assertEqual(inputs["Max lin_vel_x"].value, .6)
        inputs["Max lin_vel_x"].value = .3
        inputs["Max lin_vel_x"].callback(None)
        self.assertAlmostEqual(inputs["lin_vel_x"].min, -.2)
        self.assertAlmostEqual(inputs["lin_vel_x"].max, .3)
        self.assertEqual(inputs["lin_vel_y"].max, .3)
        self.assertEqual(inputs["ang_vel_z"].max, .5)

    def test_zero_button_resets_all_three_axes_with_fixed_axes_present(self):
        command, inputs = panel()
        inputs["lin_vel_x"].value = .2
        inputs["Zero"].callback(None)
        self.assertEqual([h.value for h in command._joystick_sliders], [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
