"""
Contains code to handle model predictions across the dataset.
"""
import pathlib
import torch

from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict

def pred_and_store(
    paths: List[pathlib.Path],
    model: torch.nn.Module,
    transform: torchvision.transforms,
    class_names: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Dict]:

    """
    Returns a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time.
    """
    pred_list = []
    for path in tqdm(paths):
        pred_dict = {}
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        start_time = timer()
        img = Image.open(path)
        transformed_image = transform(img).unsqueeze(0).to(device)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            pred_logit = model(transformed_image) # inference on target image
            pred_prob = torch.softmax(pred_logit, dim=1) # logits -> pred probs
            pred_label = torch.argmax(pred_prob, dim=1) # pred probs -> pred labels
            pred_class = class_names[pred_label.cpu()] # hardcode prediction classes to be on cpu

            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4) # dictionaries have to be on cpu for inspecting preds later on
            pred_dict["pred_class"] = pred_class

            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time - start_time, 4)

        pred_dict["correct"] = class_name == pred_class
        pred_list.append(pred_dict)

    return pred_list
