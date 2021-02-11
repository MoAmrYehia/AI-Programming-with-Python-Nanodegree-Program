import argparse
import utils

parser = argparse.ArgumentParser(description='Predict flower name from an image',)

parser.add_argument('image_path', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--top_k', action='store', default=3, type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store', default=True)

args = parser.parse_args()
print(args)

# Load the model
pre_model = utils.load_checkpoint(args.checkpoint)

# Process image
img_path = utils.open_image_path(args.image_path)
image = utils.process_image(img_path)

# Do prediction
probs, labels = utils.predict(image, pre_model, args.top_k, args.gpu)

# Get label names
cat_name = utils.load_cat_to_name(args.category_names)
labels = [cat_name[id] for id in labels]

print("Results:")
for r in range(len(labels)):
    print(f"{labels[r]} with probability {probs[r]}")