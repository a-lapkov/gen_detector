import os
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import folder_paths
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face

try:
  import torch.cuda as cuda
except:
  cuda = None

if cuda is not None:
  if cuda.is_available():
    providers = ['CUDAExecutionProvider']
  else:
    providers = ['CPUExecutionProvider']
else:
  providers = ['CPUExecutionProvider']

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, 'insightface')
insightface_models_path = os.path.join(insightface_path, 'models')

AGE_THRESHOLD_DEFAULT=15

ANALYSIS_MODELS: dict[str, Optional[FaceAnalysis]] = {
  '640': None,
  '320': None,
}

def getAnalysisModel(det_size = (640, 640)):
  global ANALYSIS_MODELS
  ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
  if ANALYSIS_MODEL is None:
    ANALYSIS_MODEL = insightface.app.FaceAnalysis(
      name='buffalo_l', providers=providers, root=insightface_path
    )
  ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
  ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
  return ANALYSIS_MODEL

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)) -> list[Face]:
  face_analyzer: FaceAnalysis = getAnalysisModel(det_size)
  return face_analyzer.get(img_data)

def convert_image(source_img: Image.Image) -> np.ndarray:
  return np.array(cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR))

def tensor_to_pil(img_tensor, batch_index=0) -> Image.Image:
  # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
  img_tensor = img_tensor[batch_index].unsqueeze(0)
  i = 255. * img_tensor.cpu().numpy()
  img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
  return img

def batch_tensor_to_pil(img_tensor) -> list[Image.Image]:
  # Convert tensor of shape [batch_size, channels, height, width] to a list of PIL Images
  return [tensor_to_pil(img_tensor, i) for i in range(img_tensor.shape[0])]

class GenderDetector:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      'required': {
        'image': ('IMAGE',),
        'child_age_threshold': ('INT', {'default': AGE_THRESHOLD_DEFAULT, 'min': 0, 'max': 0xffffffffffffffff}),
      },
      'optional': {
        'child_boy_positive': ('STRING', {'default': '', 'multiline': True}),
        'child_boy_negative': ('STRING', {'default': '', 'multiline': True}),
        'child_girl_positive': ('STRING', {'default': '', 'multiline': True}),
        'child_girl_negative': ('STRING', {'default': '', 'multiline': True}),
        'adult_man_positive': ('STRING', {'default': '', 'multiline': True}),
        'adult_man_negative': ('STRING', {'default': '', 'multiline': True}),
        'adult_woman_positive': ('STRING', {'default': '', 'multiline': True}),
        'adult_woman_negative': ('STRING', {'default': '', 'multiline': True}),
        'group_positive': ('STRING', {'default': '', 'multiline': True}),
        'group_negative': ('STRING', {'default': '', 'multiline': True}),
      }
    }

  RETURN_TYPES: tuple[str, str] = ('STRING', 'STRING', )
  RETURN_NAMES: tuple[str, str] = ('positive', 'negative', )
  FUNCTION = 'execute'
  CATEGORY = 'Magneat Suite/Gender Detector'

  def __init__(self):
    pass

  def execute(
    self,
    image,
    child_age_threshold=AGE_THRESHOLD_DEFAULT,
    child_boy_positive='',
    child_boy_negative='',
    child_girl_positive='',
    child_girl_negative='',
    adult_man_positive='',
    adult_man_negative='',
    adult_woman_positive='',
    adult_woman_negative='',
    group_positive='',
    group_negative='',
  ) -> tuple[str, str]:
    child_boy = (child_boy_positive, child_boy_negative, )
    child_girl = (child_girl_positive, child_girl_negative, )
    adult_man = (adult_man_positive, adult_man_negative, )
    adult_woman = (adult_woman_positive, adult_woman_negative, )

    pil_images = batch_tensor_to_pil(image)
    cv_image = convert_image(pil_images[0])
    faces: list[Face] = analyze_faces(cv_image)
    isGroup: bool = len(faces) > 1
    if not isGroup:
      isMan: bool = faces[0].sex == 'M'
      isChild: bool = (faces[0].age is not None and faces[0].age <= child_age_threshold)
      if isMan and isChild and '' not in child_boy:
        return child_boy
      elif not isMan and isChild and '' not in child_girl:
        return child_girl
      elif isMan and not isChild and '' not in adult_man:
        return adult_man
      elif not isMan and not isChild and '' not in adult_woman:
        return adult_woman
    return (group_positive, group_negative, )
