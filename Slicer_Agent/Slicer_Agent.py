
import logging
import os
from typing import Annotated, Optional

import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode



class Slicer_Agent(ScriptedLoadableModule):

    def __init__(self, parent):

        # Set module info
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Slicer_Agent")
        self.parent.categories = ["Automated Dental Tools" ]
        self.parent.dependencies = [] 
        self.parent.contributors = ["Paul D)"]  
        self.parent.helpText = _("""this module provides a simple template""")
        self.parent.acknowledgementText = _("""Developed by Paul """)

# Parameter node to store UI values
@parameterNodeWrapper
class Slicer_AgentParameterNode:
    msg: str = ""

# Main UI widget for the module
class Slicer_AgentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:

        # Initialize widget and logic
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:

        # Setup UI and logic
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Slicer_Agent.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.logic = Slicer_AgentLogic()
        # Observe scene events
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect button
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        # Remove observers on close
        self.removeObservers()

    def enter(self) -> None:
        # Called when module is entered
        self.initializeParameterNode()

    def exit(self) -> None:
        # Called when module is exited
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Before scene closes
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # After scene closes
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        # Ensure parameter node exists
        self.setParameterNode(self.logic.getParameterNode())


    def setParameterNode(self, inputParameterNode: Optional[Slicer_AgentParameterNode]) -> None:
        # Set and observe parameter node
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        # Enable apply button
        self.ui.applyButton.toolTip = _("Renommer les fichiers")
        self.ui.applyButton.enabled = True

    def onApplyButton(self) -> None:
        # Run logic when apply button is clicked
        with slicer.util.tryWithErrorDisplay(_("Échec du renommage des fichiers."), waitCursor=True):

            msg = self._parameterNode.msg  # <-- lire la valeur depuis le parameter node
            success = self.logic.process(msg)
            if success:
                slicer.util.messageBox("Print terminé avec succès!")

# Logic class for processing
class Slicer_AgentLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        # Get parameter node
        return Slicer_AgentParameterNode(super().getParameterNode())

    def process(self,
                msg: str) -> None:
                
        # Run CLI process with message
        print("Message: recuperer dans UI ! " + msg)
            
        # Accéder au module CLI enregistré dans Slicer
        CLI_module = slicer.modules.slicer_agent_cli
        
        # Préparer les paramètres pour le CLI
        parameters = {
            "msg": msg
        }
                       
        # Exécuter le CLI avec slicer.cli.run
        self.cliNode = slicer.cli.run(CLI_module, None, parameters)

