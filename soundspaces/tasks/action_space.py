# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import attr
import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import HabitatSimActions, HabitatSimV0ActionSpaceConfiguration
from habitat_sim.agent.controls.controls import ActuationSpec

from typing import Any, Dict, List, Optional, Type

HabitatSimActions.extend_action_space("MOVE_BACKWARD")
HabitatSimActions.extend_action_space("MOVE_LEFT")
HabitatSimActions.extend_action_space("MOVE_RIGHT")

@registry.register_action_space_configuration(name="move-all")
class MoveOnlySpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MOVE_BACKWARD: habitat_sim.ActionSpec(
                "move_backward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MOVE_RIGHT: habitat_sim.ActionSpec(
                "move_right",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MOVE_LEFT: habitat_sim.ActionSpec(
                "move_left",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            )
        }
        
   
@attr.s(auto_attribs=True, slots=True)
class QueryActuationSpec(ActuationSpec):
    # what should be the initial parameter??
    amount: float = 1.0
    
@registry.register_action_space_configuration(name="SoundspacesDialogActions-v0")
class SoundspacesDialogSimV0ActionSpaceConfiguration(
    HabitatSimV0ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("QUERY"):
            HabitatSimActions.extend_action_space("QUERY")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.QUERY: habitat_sim.ActionSpec(
                "dialog_based_navigation",
                QueryActuationSpec(
                    amount=self.config.QUERY_STEP
                ),
            )
        }
        config.update(new_config)

        return config

@registry.register_task_action
class QueryAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        return self._sim.step(HabitatSimActions.QUERY)
