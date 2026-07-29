import numpy as np
import pyvizionsdk
from pyvizionsdk.pyVizionSDK import VX_IMAGE_FORMAT

result, connected_serials = pyvizionsdk.VxDiscoverCameraDevices()

camera = pyvizionsdk.VxInitialCameraDevice(1)

result = pyvizionsdk.VxOpen(camera)
result, format_list = pyvizionsdk.VxGetFormatList(camera)

mjpg_format = None
for format in format_list:
    # get mjpg format
    if format.format == VX_IMAGE_FORMAT.VX_IMAGE_FORMAT_MJPG:
        mjpg_format = format

result = pyvizionsdk.VxSetFormat(camera, format_list[0])
# start streaming
result = pyvizionsdk.VxStartStreaming(camera)
while True:
    print(mjpg_format)
    result, image = pyvizionsdk.VxGetImage(
                camera,
                2500,
                mjpg_format,
            )
    print(result)