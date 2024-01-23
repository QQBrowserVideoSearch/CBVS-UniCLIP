import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to the test file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save inference result.",
    )
    parser.add_argument(
        "--context-length", type=int, default=52, help="The maximum length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="batch size."
    )
    parser.add_argument(
        "--input-resolution", 
        type=int, 
        default=224, 
        help="input resolution."
    )   
    parser.add_argument(
        "--ocr-presence", 
        type=int, 
        default=0, 
        help="Ocr presences or not."
    )
    parser.add_argument(
        "--ocr-semantic", 
        type=int, 
        default=0, 
        help="Ocr semantic or not."
    )    
    args = parser.parse_args()

    return args
