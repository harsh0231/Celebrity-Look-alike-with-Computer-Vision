import os
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image

def load_images(known_images_dir):
    known_encodings = []
    known_images = []

    for file in os.listdir(known_images_dir):
        filename = os.fsdecode(file)
        image = face_recognition.load_image_file(os.path.join(known_images_dir, filename))
        enc = face_recognition.face_encodings(image)
        if len(enc) > 0:
            known_encodings.append(enc[0])
            known_images.append(filename)

    return (known_encodings, known_images)

def calculate_face_distance(known_encodings, unknown_img_path, known_images, cutoff=0.5, num_results=4):
    image_to_test = face_recognition.load_image_file(unknown_img_path)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
    min_distance_index = face_distances.argmin()
    min_distance = face_distances[min_distance_index]

    if min_distance <= cutoff:
        return (unknown_img_path, known_images[min_distance_index])
    else:
        return (unknown_img_path, "Unknown")

known_encodings, known_images = load_images("/cxldata/projects/lookalikeceleb/images")

original_image_path = "../lookalikeceleb/myimage.jpg"
matching_image = calculate_face_distance(known_encodings, original_image_path, known_images)[1]

img_1 = mpimg.imread(original_image_path)
img_2 = mpimg.imread('/cxldata/projects/lookalikeceleb/images/' + matching_image)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_1)
ax[0].axis('off')  # Turn off axis
ax[1].imshow(img_2)
ax[1].axis('off')  # Turn off axis

print('Hey, you look like ' + os.path.splitext(matching_image)[0] + '!')

plt.show()
