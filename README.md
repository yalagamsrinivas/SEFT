Requirements

Run the following command to install requirements:

pip install -r requirements.txt

Training

To train the 4 seft models from the scratch, run this command with a model argument:

bash train.sh seft_v1

bash train.sh seft_v2

bash train.sh seft_v3

bash train.sh seft_v4

or 

run train_main.py with below arguments 

--paths_text

text/tweet

--paths_technical

technical

--model_name

seft_v4

