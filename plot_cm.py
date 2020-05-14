import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from main import *


# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('--trained_model_path', type=str, required=True)
parser.add_argument('--graphs_path', type=str, required=True)
parser.add_argument('--saving', type=int, required=True)
parser.add_argument('--model_type', type=str, required=True)
args = parser.parse_args()

TRAINED_MODEL_PATH = args.trained_model_path 
MODEL_TYPE = args.model_type
CM_PATH = f"{args.graphs_path}/CM_{MODEL_TYPE}"
SAVING = args.saving

test_df = pd.read_csv(TEST_LABELS_PATH)

BATCH_SIZE = 1
SR = SAMPLE_RATE
AUDIO_DURATION = 10
NB_SPECIES = len(set(test_df['label']))
print("NB_SPECIES: ", NB_SPECIES)

RANDOM_STATE = 17
random.seed(RANDOM_STATE)


# Load Model
criterion = nn.CrossEntropyLoss()
print("Loading model...")
model = load_model(model_type=MODEL_TYPE, sample_rate=SR, nb_species=NB_SPECIES, 
                   model_path=TRAINED_MODEL_PATH, after_train=True)

# Load testset
print("Processing dataset")
testloader = process_data(df=test_df, batch_size=BATCH_SIZE, 
                          sample_rate=SR, audio_duration=AUDIO_DURATION, 
                          random_state=RANDOM_STATE, do_plot=False)


history_training = {}
dataloaders = {'test': testloader[0]}
dataset_sizes = {'test': testloader[1]}

history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                              dataloaders=dataloaders, dataset_sizes=dataset_sizes)


y_pred = [y.cpu() for y in history_training['y_pred']]
y_true = [y.cpu() for y in history_training['y_true']]

# Classification report
accuracy = round(accuracy_score(y_true, y_pred)*100, 3)
mse = round(mean_squared_error(y_true, y_pred), 3)
print(f'Accuracy: {accuracy}%')
print(f'MSE: {mse}')
target_names = [f'class {i+1}' for i in range(NB_SPECIES)]
print(classification_report(y_true, y_pred, target_names=target_names))


# Plot and save Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index = [i+1 for i in range(NB_SPECIES)], 
                     columns = [i+1 for i in range(NB_SPECIES)])
plt.figure(figsize = (10,7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(df_cm, cmap=cmap, annot=True)
plt.title(f"Confusion Matrix for {MODEL_TYPE}")

if SAVING:
    plt.savefig(CM_PATH)
    print(f"Confusion Matrix saved at {CM_PATH}")

plt.show()

