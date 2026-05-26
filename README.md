All Code Links are provided here
Finetune SpeechGPT
Speech-7B-cm is a foundational model with strong alignment between speech and text. We encourage fine-tuning SpeechGPT based on this model.

>Step1: prepare your data following the format in SpeechInstruct Cross-modal Instruction set.

>Step2: download SpeechGPT-7B-cm locally.

>Step3: Modify the METAROOT, DATAROOT, and OUTROOT parameters in the scripts/cm_sft.sh script to yours and then run it. For LoRA fine-tuning, update the METAROOT, DATAROOT, and OUTROOT parameters in the scripts/com_sft.sh script and run it.

Link is provided here-https://huggingface.co/OpenMOSS-Team/SpeechGPT-2.0-preview-7B/blob/main/README.md

#Code is available here-https://github.com/0nutation/SpeechGPT/tree/main/speechgpt#finetune-speechgpt

----------------------------------------------------------------------------------------------------------------
>MOS Calculation
>https://github.com/sarulab-speech/UTMOS22

>git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
>cd UTMOS-demo
>pip install -r requirements.txt

>python predict.py --mode predict_dir --inp_dir /path/to/wav/dir/ --bs <batchsize> --out_path /path/to/csv/file.csv
------------------------------------------------------------------------------------------------------------------
https://github.com/tiantiaf0627/vox-profile-release
'''
Load libraries
>import torch
>import torch.nn.functional as F
>from src.model.accent.whisper_accent import WhisperWrapper


>english_accent_list = [
    'East Asia', 'English', 'Germanic', 'Irish', 
    'North America', 'Northern Irish', 'Oceania', 
    'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
    'South African', 'Southeast Asia', 'South Asia', 'Welsh'
]
    
Find device
>device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

Load model from Huggingface

>whisper_model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-narrow-accent").to(device)
>whisper_model.eval()

Load data, here just zeros as the example
Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel

>max_audio_length = 15 * 16000

>data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]

>whisper_logits, whisper_embeddings = whisper_model(data, return_feature=True)
    
Probability and output

>whisper_prob = F.softmax(whisper_logits, dim=1)

>print(english_accent_list[torch.argmax(whisper_prob).detach().cpu().item()])

-------------------------------------------------------------------------------------------------------------------------------------------------

For Generating dataset 3, first we run preprocessing.py to preproces the available data, then we run the encoder to generate the speaker embeddings, which is passed through decoder to generate mel spectograms,

