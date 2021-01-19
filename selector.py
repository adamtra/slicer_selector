import sys
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import sitkUtils
import numpy as np
from subprocess import check_output

#
# selector
#

class selector(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "selector" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Projekt"]
    self.parent.dependencies = []
    self.parent.contributors = ["Adam Trawinski"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# selectorWidget
#

class selectorWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)


    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.inputSelector.setToolTip("Pick the input to the algorithm.")
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # iterations value
    #
    self.iterationsSliderWidget = ctk.ctkSliderWidget()
    self.iterationsSliderWidget.singleStep = 1
    self.iterationsSliderWidget.minimum = 1
    self.iterationsSliderWidget.maximum = 200
    self.iterationsSliderWidget.value = 20
    self.iterationsSliderWidget.setToolTip(
        """Number of iterations to run.""")
    parametersFormLayout.addRow("Iterations", self.iterationsSliderWidget)
    
    #
    # smoothing value
    #
    self.smoothingSliderWidget = ctk.ctkSliderWidget()
    self.smoothingSliderWidget.singleStep = 1
    self.smoothingSliderWidget.minimum = 1
    self.smoothingSliderWidget.maximum = 4
    self.smoothingSliderWidget.value = 1
    self.smoothingSliderWidget.setToolTip(
        """Number of times the smoothing operator is applied per iteration.
            Larger values lead to smoother segmentations.""")
    parametersFormLayout.addRow("Smoothing", self.smoothingSliderWidget)
    
    #
    # threshold value
    #
    self.thresholdSliderWidget = ctk.ctkSliderWidget()
    self.thresholdSliderWidget.singleStep = 0.01
    self.thresholdSliderWidget.minimum = 0
    self.thresholdSliderWidget.maximum = 1
    self.thresholdSliderWidget.value = 0.5
    self.thresholdSliderWidget.setToolTip("""Areas of the image with a value smaller than this threshold will be
                                               considered borders. The evolution of the contour will stop in this
                                               areas.""")
    parametersFormLayout.addRow("Threshold (GAC only)", self.thresholdSliderWidget)
    

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableBaloonFlagCheckBox = qt.QCheckBox()
    self.enableBaloonFlagCheckBox.checked = 1
    self.enableBaloonFlagCheckBox.setToolTip("""Balloon force to guide the contour in non-informative areas of the
                                                  image, i.e., areas where the gradient of the image is too small to push
                                                  the contour towards a border. A negative value will shrink the contour,
                                                  while a positive value will expand the contour in these areas.""")
    parametersFormLayout.addRow("Baloon (GAC only)", self.enableBaloonFlagCheckBox)


    self.markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")

    self.placeWidget = slicer.qSlicerMarkupsPlaceWidget()
    self.placeWidget.setMRMLScene(slicer.mrmlScene)
    self.placeWidget.setCurrentNode(self.markupsNode)
    self.placeWidget.buttonsVisible = False
    self.placeWidget.placeButton().show()
    self.placeWidget.connect(
        'activeMarkupsFiducialPlaceModeChanged(bool)', self.placementModeChanged)
    parametersFormLayout.addRow(self.placeWidget)

    #
    # name value
    #
    self.textWidget = qt.QPlainTextEdit("Tumor")
    parametersFormLayout.addRow("Segment name", self.textWidget)

    #
    # color value
    #
    self.color = [255, 255, 0]
    self.colorWidget = qt.QColorDialog()
    self.colorWidget.setOption(2)
    self.colorWidget.setCurrentColor(qt.QColor(self.color[0], self.color[1], self.color[2]))
    self.colorWidget.connect(
        'currentColorChanged(QColor)', self.colorChanged)
    parametersFormLayout.addRow("Select color", self.colorWidget)


    #
    # 3D - MorphACWE Button
    #
    self.acwe3d = qt.QPushButton("3D - MorphACWE")
    self.acwe3d.toolTip = "Run the algorithm."
    self.acwe3d.enabled = True
    self.acwe3d.connect('clicked(bool)', self.onAcwe3d)
    parametersFormLayout.addRow(self.acwe3d)
    #
    # 2D - MorphACWE Button
    #
    self.acwe2d = qt.QPushButton("2D - MorphACWE")
    self.acwe2d.toolTip = "Run the algorithm."
    self.acwe2d.enabled = True
    self.acwe2d.connect('clicked(bool)', self.onAcwe2d)
    parametersFormLayout.addRow(self.acwe2d)

    #
    # 2D - MorphACWE Prev Button
    #
    self.acwe_prev2d = qt.QPushButton("2D - MorphACWE (prev)")
    self.acwe_prev2d.toolTip = "Run the algorithm."
    self.acwe_prev2d.enabled = True
    self.acwe_prev2d.connect('clicked(bool)', self.onAcwe2dPrev)
    parametersFormLayout.addRow(self.acwe_prev2d)

    #
    # 3D - MorphGAC Button
    #
    self.gac3d = qt.QPushButton("3D - MorphGAC")
    self.gac3d.toolTip = "Run the algorithm."
    self.gac3d.enabled = True
    self.gac3d.connect('clicked(bool)', self.onGac3d)
    parametersFormLayout.addRow(self.gac3d)

    #
    # 2D - MorphGAC Button
    #
    self.gac2d = qt.QPushButton("2D - MorphGAC")
    self.gac2d.toolTip = "Run the algorithm."
    self.gac2d.enabled = True
    self.gac2d.connect('clicked(bool)', self.onGac2d)
    parametersFormLayout.addRow(self.gac2d)


    # Add vertical spacer
    self.layout.addStretch(1)

    self.ras = [0, 0, 0]

  def placementModeChanged(self, active):
    if active == False:
      slicer.mrmlScene.Redo()
      self.ras = [0, 0, 0]
      slicer.util.getNode('Crosshair').GetCursorPositionRAS(self.ras)

  def cleanup(self):
    pass

  def onAcwe3d(self):
    self.onApplyButton(0)
  
  def onAcwe2d(self):
    self.onApplyButton(1)

  def onGac3d(self):
    self.onApplyButton(2)

  def onGac2d(self):
    self.onApplyButton(3)

  def onAcwe2dPrev(self):
    self.onApplyButton(4)

  def colorChanged(self, color):
      color = qt.QColor(color)
      self.color[0] = color.red()
      self.color[1] = color.green()
      self.color[2] = color.blue()

  def onApplyButton(self, mode):
    logic = selectorLogic()
    enableBaloonFlag = self.enableBaloonFlagCheckBox.checked
    iterations = self.iterationsSliderWidget.value
    smoothing = self.smoothingSliderWidget.value
    threshold = self.thresholdSliderWidget.value
    volume = self.inputSelector.currentNode()
    
    name = self.textWidget.toPlainText()
    numFids = self.markupsNode.GetNumberOfFiducials()
    for i in range(numFids):
        self.markupsNode.SetNthFiducialVisibility(i, 0)

    logic.run(mode, volume, self.ras, enableBaloonFlag,
              iterations, smoothing, threshold, self.color, name)


#
# selectorLogic
#


class selectorLogic(ScriptedLoadableModuleLogic):

  def calc_coord(self, volumeNode, coord):
    volumeIjkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(volumeIjkToRas)
    point_VolumeRas = [0, 0, 0, 1]
    volumeIjkToRas.MultiplyPoint(np.append(coord, 1.0), point_VolumeRas)
    point_VolumeRas.pop()
    return point_VolumeRas

  def run(self, mode, volumeNode, ras, enableBaloonFlag, iterations, smoothing, threshold, color, name):
    """
    Run the actual algorithm
    """
    print(ras)
    logging.info('Processing started')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if enableBaloonFlag == True:
      ballon = 1
    else:
      ballon = -1

    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(np.append(ras, 1.0), point_Ijk)
    point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]
    # Print output
    data = slicer.util.arrayFromVolume(volumeNode)
    np.save(dir_path + '/utils/image.npy', data)
    np.save(dir_path + '/utils/coord.npy', point_Ijk)

    print('Snake')
    command_line = ["/usr/bin/python3",
                    dir_path + "/utils/snake.py",
                    str(int(mode)),
                    str(int(float(iterations))),
                    str(int(float(smoothing))),
                    str(threshold),
                    str(ballon)]
    command_result = check_output(
        command_line, env=slicer.util.startupEnvironment())
    print(command_result)

    
    data = np.load(dir_path + '/utils/out.npy')
    coord = np.load(dir_path + '/utils/coord.npy')
    new_data = []

    for i, val_i in enumerate(data):
      for j, val_j in enumerate(data[i]):
        line_started = False
        line = vtk.vtkLineSource()
        for k, val_k in enumerate(data[i][j]):
          if val_k != 0 and line_started == False:
            line_started = True
            line.SetPoint1(self.calc_coord(volumeNode, [k, j, i]))
            line.Update()
          if val_k == 0 and line_started == True:
            line.SetPoint2(self.calc_coord(volumeNode, [k - 1, j, i]))
            line.Update()
            new_data.append(line)
            line_started = False
            line = vtk.vtkLineSource()


    segmentationNode = slicer.vtkMRMLSegmentationNode()
    slicer.mrmlScene.AddNode(segmentationNode)
    segmentationNode.CreateDefaultDisplayNodes()
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    append = vtk.vtkAppendPolyData()
    for line in new_data:
      append.AddInputData(line.GetOutput())

    append.Update()

    red = color[0] / 255.0
    green = color[1] / 255.0
    blue = color[2] / 255.0
    tempSegment = segmentationNode.AddSegmentFromClosedSurfaceRepresentation(append.GetOutput(), "Temp", [red, green, blue])

    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode)

    labelmapVolumeNode.SetName(name)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)
    segmentationNode.CreateClosedSurfaceRepresentation()
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
    segmentationNode.RemoveSegment(tempSegment)


    print('Zrobione')

    return True


class selectorTest(ScriptedLoadableModuleTest):

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    return True
