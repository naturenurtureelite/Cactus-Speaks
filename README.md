All Code Links are provided here
Finetune SpeechGPT
Speech-7B-cm is a foundational model with strong alignment between speech and text. We encourage fine-tuning SpeechGPT based on this model.

Step1: prepare your data following the format in SpeechInstruct Cross-modal Instruction set.

Step2: download SpeechGPT-7B-cm locally.

Step3: Modify the METAROOT, DATAROOT, and OUTROOT parameters in the scripts/cm_sft.sh script to yours and then run it. For LoRA fine-tuning, update the METAROOT, DATAROOT, and OUTROOT parameters in the scripts/com_sft.sh script and run it.

Link is provided here-https://huggingface.co/OpenMOSS-Team/SpeechGPT-2.0-preview-7B/blob/main/README.md

Code is available here-https://github.com/0nutation/SpeechGPT/tree/main/speechgpt#finetune-speechgpt
