import os
import shutil
from typing import Annotated
import urllib.request
import vtk
import slicer
import qt
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer import vtkMRMLScalarVolumeNode
import importlib

import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("SurgMovPred")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Library dependency management
def check_lib_installed(import_name: str) -> bool:
    """
    Silently checks if a Python library is installed and accessible.
    'import_name' is the name used in the code (e.g., 'llama_cpp').
    """
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False
    
def install_function(list_libs: list) -> None:
    """
    Installs a list of packages via pip in the 3D Slicer environment.
    Assumes the user has already given permission.
    """

    original_cc = os.environ.get("CC")
    original_cxx = os.environ.get("CXX")

    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"

    for lib in list_libs:
        slicer.util.showStatusMessage(f"Installing {lib}... Please wait.")
        slicer.app.processEvents()

        try:
            if lib == "llama-cpp-python":
                slicer.util.pip_install("llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
            else:
                slicer.util.pip_install(lib)

            slicer.util.showStatusMessage(f"{lib} successfully installed!", 3000)
            logger.info(f"Successfully installed {lib}")

        except Exception as e:
            logger.error(f"Failed to install {lib}: {str(e)}")
            slicer.util.errorDisplay(f"Failed to install {lib}.\nError: {str(e)}")

    if original_cc is not None:
        os.environ["CC"] = original_cc
    else:
        del os.environ["CC"]

    if original_cxx is not None:
        os.environ["CXX"] = original_cxx
    else:
        del os.environ["CXX"]

def check_dependencies() -> bool:
    """
    Checks dependencies when the Apply button is clicked.
    Returns True if everything is ready, False if it should be cancelled.
    Attempts up to 2 verifications with proper logging.
    """
    max_retries = 1
    for attempt in range(max_retries + 1):
        missing_libs = []

        if not check_lib_installed("llama_cpp"):
            missing_libs.append("llama-cpp-python")

        if not missing_libs:
            logger.info("All dependencies verified and available.")
            return True

        if attempt < max_retries:
            libs_str = "\n".join([f"- {lib}" for lib in missing_libs])

            msg = (
                "The SurgMovPred module requires the following libraries to function:\n\n"
                f"{libs_str}\n\n"
                "Do you agree to modify Slicer's environment to install them? "
                "This may take a few minutes."
            )

            if slicer.util.confirmOkCancelDisplay(msg):
                logger.info(f"Installing missing dependencies: {missing_libs}")
                install_function(missing_libs)
                slicer.app.processEvents()
            else:
                logger.warning("Installation cancelled by user.")
                slicer.util.warningDisplay("Installation cancelled. Extraction has been stopped.")
                return False
        else:
            logger.error(f"Failed to install required dependencies: {missing_libs}")
            error_msg = (
                "Failed to install required dependencies:\n\n"
                f"{libs_str}\n\n"
                "Please restart Slicer and try again."
            )
            slicer.util.errorDisplay(error_msg)
            return False

    return False

class SurgMovPred(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SurgMovPred")
        self.parent.categories = ["Automated Dental Tools" ]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Paul Dumont, University of North Carolina, Chapell Hill"]  
        self.parent.helpText = _("""
        This tool helps to create summaries of clinical notes. 
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)


#
# SurgMovPredParameterNode
#
@parameterNodeWrapper
class SurgMovPredParameterNode:
    """
    Parameters for Clinical Notes Extraction UI.

    notesFolder_input - Folder containing clinical notes (.docx/.pdf/.txt).
    modelType - Model type selection: 'Mini' (Light/Fast) or 'Max' (Heavy/Precise).
    notesType - Notes type selection: 'TMJ' or 'Ortho'.
    notesFolder_output - Folder for summary output.
    """

    inputFolder: str = ""
    modelPath: str = ""
    outputFolder: str = ""

#
# SurgMovPredWidget
#
class SurgMovPredWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)


        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SurgMovPred.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create logic class.
        self.logic = SurgMovPredLogic()

        isDarkMode = self._isDarkMode()
        styleSheet = self._getStyleSheet(isDarkMode)
        uiWidget.setStyleSheet(styleSheet)
        
        # Also apply label-specific stylesheet
        self._applyLabelStyleSheets(isDarkMode)
        self._applyButtonStyleSheets(isDarkMode)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.cancelButton.connect("clicked(bool)", self.onCancelCliButton)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.downloadTestFilesButton.connect("clicked(bool)", self.onRunTestFilesButton)
        self.ui.DefaultModelButton.connect("clicked(bool)", self.onRunTestFilesButton)

        self.ui.inputFolderLineEdit.currentPathChanged.connect(self._checkCanApply)
        self.ui.modelFolderLineEdit.currentPathChanged.connect(self._checkCanApply)
        self.ui.outputFolderLineEdit.currentPathChanged.connect(self._checkCanApply)

        self.ui.cancelButton.setVisible(False)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.inputFolder:
            self._parameterNode.inputFolder = ""

        if not self._parameterNode.modelPath:
            self._parameterNode.modelPath = ""
        
        if not self._parameterNode.outputFolder:
            self._parameterNode.outputFolder = ""

    def setParameterNode(self, inputParameterNode: SurgMovPredParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """Enable/disable the apply button based on required fields."""
        if not self._parameterNode:
            self.ui.applyButton.enabled = False
            return
        
        self._parameterNode.inputFolder = self.ui.inputFolderLineEdit.currentPath
        self._parameterNode.modelPath = self.ui.modelFolderLineEdit.currentPath
        self._parameterNode.outputFolder = self.ui.outputFolderLineEdit.currentPath

        if self._parameterNode.inputFolder != "" and self._parameterNode.outputFolder != "" and self._parameterNode.modelPath != "":
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.enabled = False

    def onRunTestFilesButton(self) -> None:
        """Run test files when user clicks 'Run Test Files' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to download test files."), waitCursor=True):
            # Get the selected notes type from parameter node
            notesType = self._parameterNode.notesType
            
            # Copy test files and get the paths
            input_path, output_path = self.logic.copyTestFiles(notesType)
            
            # Update the folder paths in the UI
            self.ui.notesFolderLineEdit_input.currentPath = input_path
            self.ui.notesFolderLineEdit_output.currentPath = output_path


    def onApplyButton(self) -> None:
        """Run processing when user clicks Apply button."""

        if not check_dependencies():
            return

        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):

            inputFolder = self._parameterNode.inputFolder
            modelPath = self._parameterNode.modelPath
            outputFolder = self._parameterNode.outputFolder

            logger.info(f"Input folder   : {inputFolder}")
            logger.info(f"Output folder  : {outputFolder}")
            logger.info(f"Selected model : {modelPath}")


            cliNode = self.logic.process(inputFolder,modelPath,outputFolder)

            if cliNode:
                self.ui.cancelButton.setVisible(True)
                self.addObserver(cliNode, slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliFinished)


    def onCancelCliButton(self) -> None:
        """Cancel the running CLI process."""
        if self.logic and hasattr(self.logic, 'cliNode') and self.logic.cliNode:
            self.logic.cliNode.Cancel()
            self.cliProgressBar.visible = False
            self.cliCancelButton.visible = False
            slicer.util.warningDisplay("Processing cancelled by user.")

    def onCliFinished(self, caller, event) -> None:
        """Hide progress bar and cancel button when CLI finishes."""
        status = caller.GetStatus()
        if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
            self.ui.cancelButton.setVisible(False)

    def _isDarkMode(self) -> bool:
        """Check if the application is in dark mode."""
        try:
            palette = slicer.app.palette()
            bgColor = palette.color(qt.QPalette.Window)
            luminance = (0.299 * bgColor.red() + 0.587 * bgColor.green() + 0.114 * bgColor.blue()) / 255.0
            return luminance < 0.5
        except:
            return False
        
    def _getStyleSheet(self, isDarkMode: bool) -> str:
        """Generate stylesheet based on theme."""
        if isDarkMode:
            return """
            qMRMLWidget {
              background-color: #2b2b2b;
            }
            ctkCollapsibleButton {
              background-color: #383838;
              border: 1px solid #454545;
              border-radius: 6px;
              margin-bottom: 8px;
              font-weight: 600;
              padding: 6px 10px;
              color: #e0e0e0;
            }
            ctkCollapsibleButton:hover {
              border: 1px solid #3498db;
              background-color: #414141;
            }
            QLineEdit, QTextEdit {
              background-color: #353535;
              border: 1px solid #454545;
              border-radius: 4px;
              padding: 6px;
              color: #e0e0e0;
              selection-background-color: #3498db;
            }
            QLineEdit:focus, QTextEdit:focus {
              border: 2px solid #3498db;
              background-color: #383838;
            }
            QComboBox {
              background-color: #353535;
              border: 1px solid #454545;
              border-radius: 4px;
              padding: 4px 6px;
              color: #e0e0e0;
            }
            QComboBox:focus {
              border: 2px solid #3498db;
            }
            QComboBox::drop-down {
              width: 20px;
              border: none;
            }
            QComboBox QAbstractItemView {
              background-color: #353535;
              color: #e0e0e0;
              selection-background-color: #3498db;
              border: 1px solid #454545;
            }
            QProgressBar {
              border: 1px solid #454545;
              border-radius: 4px;
              background-color: #353535;
              padding: 2px;
              color: #e0e0e0;
            }
            QProgressBar::chunk {
              background-color: #3498db;
              border-radius: 3px;
            }
            """
        else:
            return """
            qMRMLWidget {
              background-color: #f8f9fa;
            }
            ctkCollapsibleButton {
              background-color: #ffffff;
              border: 1px solid #e0e6ed;
              border-radius: 6px;
              margin-bottom: 8px;
              font-weight: 600;
              padding: 6px 10px;
              color: #2c3e50;
            }
            ctkCollapsibleButton:hover {
              border: 1px solid #3498db;
              background-color: #fbfcfd;
            }
            QLineEdit, QTextEdit {
              background-color: #ffffff;
              border: 1px solid #e0e6ed;
              border-radius: 4px;
              padding: 6px;
              color: #2c3e50;
              selection-background-color: #3498db;
            }
            QLineEdit:focus, QTextEdit:focus {
              border: 2px solid #3498db;
            }
            QComboBox {
              background-color: #ffffff;
              border: 1px solid #e0e6ed;
              border-radius: 4px;
              padding: 4px 6px;
              color: #2c3e50;
            }
            QComboBox:focus {
              border: 2px solid #3498db;
            }
            QComboBox::drop-down {
              width: 20px;
              border: none;
            }
            QComboBox QAbstractItemView {
              background-color: #ffffff;
              color: #2c3e50;
              selection-background-color: #3498db;
              border: 1px solid #e0e6ed;
            }
            QProgressBar {
              border: 1px solid #e0e6ed;
              border-radius: 4px;
              background-color: #ffffff;
              padding: 2px;
              color: #2c3e50;
            }
            QProgressBar::chunk {
              background-color: #3498db;
              border-radius: 3px;
            }
            """

    def _applyLabelStyleSheets(self, isDarkMode: bool) -> None:
        """Apply label-specific stylesheets."""
        if isDarkMode:
            labelStyle = "color: #b0b0b0; font-weight: 600;"
        else:
            labelStyle = "color: #34495e; font-weight: 600;"
        
        # List of labels to style
        labels = [
            'label_5', 'label_4', 'label_2', 'label_6', 'label_7', 'label_3', 'label', 'modeLabel', 't2label', 'excellabel'
        ]
        
        for labelName in labels:
            if hasattr(self.ui, labelName):
                label = getattr(self.ui, labelName)
                label.setStyleSheet(labelStyle)

    def _applyButtonStyleSheets(self, isDarkMode: bool) -> None:
        """Apply button-specific stylesheets."""
        if isDarkMode:
            # Dark mode button styles
            standardButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ba3ff, stop:1 #3498db);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb3ff, stop:1 #2980b9);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:disabled {
              background-color: #555555;
              color: #888888;
            }
            """
            
            cancelButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #a93226);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a93226, stop:1 #922b21);
            }
            QPushButton:disabled {
              background-color: #555555;
              color: #888888;
            }
            """
        else:
            # Light mode button styles
            standardButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ba3ff, stop:1 #3498db);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb3ff, stop:1 #2980b9);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:disabled {
              background-color: #bdc3c7;
              color: #95a5a6;
            }
            """
            
            cancelButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #a93226);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a93226, stop:1 #922b21);
            }
            QPushButton:disabled {
              background-color: #bdc3c7;
              color: #95a5a6;
            }
            """
        
        # Apply standard style to most buttons
        for buttonName in ['applyButton','DefaultModelButton','downloadTestFilesButton']:
            if hasattr(self.ui, buttonName):
                button = getattr(self.ui, buttonName)
                button.setStyleSheet(standardButtonStyle)
        
        # Apply cancel style to cancel button
        if hasattr(self.ui, 'cancelButton'):
            self.ui.cancelButton.setStyleSheet(cancelButtonStyle)




#
# SurgMovPredLogic
#
class SurgMovPredLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self,):
        return SurgMovPredParameterNode(super().getParameterNode())
    
    def copyTestFiles(self, notesType: str) -> tuple:
        """Copy test files for the selected notes type from Resources/testfiles to SlicerDownloads/CNE/testfiles/{notesType}.
        
        Args:
            notesType: Either 'TMJ' or 'Ortho' to specify which test files to copy
            
        Returns:
            tuple: (input_folder_path, output_folder_path)
        """
        # Get the path to the testfiles directory (relative to this module)
        moduleDir = os.path.dirname(__file__)
        sourceTestFilesPath = os.path.join(moduleDir, "Resources", "testfiles")
        
        if not os.path.exists(sourceTestFilesPath):
            raise FileNotFoundError(f"Test files directory not found: {sourceTestFilesPath}")
        
        # Define destination path in SlicerDownloads/CNE/testfiles/{notesType}
        documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
        destBasePath = os.path.join(
            documents,
            slicer.app.applicationName + "Downloads",
            "CNE",
            "testfiles",
            notesType
        )
        
        # Create destination directory if it doesn't exist
        if not os.path.exists(destBasePath):
            os.makedirs(destBasePath)
        
        # Determine which folders to copy based on notesType
        if notesType == "Ortho":
            folders_to_copy = ["input_Ortho", "output_Ortho"]
        elif notesType == "TMJ":
            folders_to_copy = ["input_TMJ", "output_TMJ"]
        else:
            raise ValueError(f"Unknown notes type: {notesType}")
        
        # Copy only the relevant folders
        for folder_name in folders_to_copy:
            source_folder = os.path.join(sourceTestFilesPath, folder_name)
            dest_folder = os.path.join(destBasePath, folder_name)
            
            if not os.path.exists(source_folder):
                logger.warning(f"Source folder not found: {source_folder}")
                continue
            
            # Remove destination if it already exists
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
            
            # Copy the folder
            shutil.copytree(source_folder, dest_folder)
            logger.info(f"Test folder copied from {source_folder} to {dest_folder}")
            logger.info(f"Test folder download: {dest_folder}")
        
        # Determine input and output paths
        if notesType == "Ortho":
            input_folder = "input_Ortho"
            output_folder = "output_Ortho"
        else:  # TMJ
            input_folder = "input_TMJ"
            output_folder = "output_TMJ"
        
        input_path = os.path.join(destBasePath, input_folder)
        output_path = os.path.join(destBasePath, output_folder)
        
        return input_path, output_path
    
    def getModelPath(self, modelType: str,notesType: str):
        """Returns the local path to the model, downloading it if necessary with a progress popup."""

        # 1. Configuration of the model based on UI selection
        if notesType == "Ortho":
            if modelType == "Mini":

                repo_id = "dcbia/Phi-3.5-Mini-Instruct-Ortho"
                fileName = "model-q4_0.gguf" 
                localModelName = "Phi-3.5-Mini-Ortho.gguf"
                dialogText = "Downloading Mini Ortho AI model (approx. 2.4 GB)..."
                
            elif modelType == "Max":
                repo_id = "dcbia/Meta-Llama-3.1-8B-Instruct-Ortho"
                fileName = "model-q4_0.gguf" 
                localModelName = "Meta-Llama-3.1-8B-Ortho.gguf"
                dialogText = "Downloading Max Ortho AI model (approx. 4.7 GB)..."

        # 1. Configuration of the model based on UI selection
        elif notesType == "TMJ":
            if modelType == "Mini":
                repo_id = "dcbia/Qwen-2.5-1.5B-Instruct-TMJ"
                fileName = "Qwen-2.5-1.5B-Instruct-TMJ-q4_0.gguf" 
                localModelName = "Qwen-2.5-1.5B-TMJ.gguf"
                dialogText = "Downloading Mini TMJ AI model (approx. 1 GB)..."
                
            elif modelType == "Max":
                repo_id = "dcbia/Qwen-2.5-7B-Instruct-TMJ"
                fileName = "Qwen-2.5-7B-Instruct-TMJ-q4_0.gguf" 
                localModelName = "Qwen-2.5-7B-TMJ.gguf"
                dialogText = "Downloading Max TMJ AI model (approx. 4.4 GB)..."
        
        else:
            raise ValueError(f"Unknown model type selected: {modelType}")

        modelUrl = f"https://huggingface.co/{repo_id}/resolve/main/{fileName}"
        
        # 2. Directory structure
        documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
        SlicerDownloadPath = os.path.join(
            documents,
            slicer.app.applicationName + "Downloads",
            "CNE",
            "model"
        )
        
        if not os.path.exists(SlicerDownloadPath):
            os.makedirs(SlicerDownloadPath)
            
        destPath = os.path.join(SlicerDownloadPath, localModelName)

        # 3. Check and download
        if not os.path.exists(destPath):
            logger.info(f"Downloading {modelType} model to: {destPath}")
            
            # --- Create the popup (QProgressDialog) ---
            progressDialog = qt.QProgressDialog(dialogText, "Cancel", 0, 100)
            progressDialog.setWindowTitle(f"CNE - Preparing {modelType} AI Model")
            progressDialog.setWindowModality(qt.Qt.WindowModal) 
            progressDialog.setMinimumDuration(0)
            progressDialog.show()

            # --- Callback function to update the popup ---
            def download_progress(count, block_size, total_size):
                if progressDialog.wasCanceled:
                    raise Exception("Download cancelled by user.")
                
                if total_size > 0:
                    percent = min(int((count * block_size * 100) / total_size), 100)
                    progressDialog.setValue(percent)
                
                # Forces Slicer to refresh the UI (prevents freezing)
                slicer.app.processEvents()
            
            # --- Start the download ---
            import urllib.request
            try:
                urllib.request.urlretrieve(modelUrl, destPath, reporthook=download_progress)
                progressDialog.setValue(100)
                slicer.util.showStatusMessage(f"{modelType} model download completed!", 3000)
                
            except Exception as e:
                if os.path.exists(destPath):
                    os.remove(destPath)
                slicer.util.errorDisplay(f"Download failed or was cancelled: {e}")
                progressDialog.close()
                raise e 
                
            finally:
                progressDialog.close()
            
        return destPath

    def process(self, notesFolder_input: str, modelPath: str, notesFolder_output: str) -> bool:
        """Process clinical notes using the selected model and parameters."""

        if not notesFolder_input or not modelPath or not notesFolder_output:
            missing = []
            if not notesFolder_input:
                missing.append("Input folder")
            if not modelPath:
                missing.append("Model Path")
            if not notesFolder_output:
                missing.append("Output folder")

            error_msg = f"Process cancelled: Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            slicer.util.errorDisplay(error_msg)
            return None

        os.makedirs(notesFolder_output, exist_ok=True)

        CLI_module = slicer.modules.surgmovpred_cli
        parameters = {
            "inputFolder": notesFolder_input,
            "modelPath": modelPath,
            "outputFolder": notesFolder_output,
        }

        logger.info(f"Launching CLI")
        self.cliNode = slicer.cli.run(CLI_module, None, parameters)
        self.cliNode.AddObserver(slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliModified)

        return self.cliNode

    def onCliProgress(self, caller, event):
        """Callback triggered on CLI progress updates."""
        progress = caller.GetProgress()

    def onCliModified(self, caller, event):
        """Callback triggered when CLI status changes (completed, cancelled, etc.)."""
        status = caller.GetStatus()

        if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
            logger.info("Background process finished (CLI)")

            if status == slicer.vtkMRMLCommandLineModuleNode.Completed:
                logger.info("SurgMovPred- COMPLETE")
            elif status == slicer.vtkMRMLCommandLineModuleNode.Cancelled:
                logger.info("PROCESS CANCELLED BY USER")

            output_text = caller.GetOutputText()
            if output_text:
                logger.info("--- Detailed CLI Logs ---")
                logger.info(output_text.strip())
                logger.info("---------------------------\n")

            error_text = caller.GetErrorText()
            if error_text:
                logger.error("--- CLI ERRORS ---")
                logger.error(error_text.strip())
                logger.error("---------------------\n")
