Celebrity Look-Alike Finder

This Python script uses the face_recognition library to identify and compare facial features between a user-uploaded image and a dataset of known celebrity images. The program processes an image directory to create face encodings, and then calculates the facial distance between these encodings and the encoding of an unknown image to find the closest match.

Features
Image loading and face encoding using face_recognition.
Matching function to compare unknown image against known encodings.
Display comparison results visually using matplotlib.
How It Works
load_images() scans a directory of images, encoding each face using face_recognition and stores them.
calculate_face_distance() computes the similarity between the unknown face and our dataset, returning the closest match if it falls below a similarity threshold.
The script displays the uploaded image and its celebrity look-alike side by side in a matplotlib plot and prints out the celebrity match.
Output
When a match is found, the program displays both the user's photo and the celebrity look-alike's photo in a two-panel figure, with axes turned off for a cleaner look. It also prints a message to the console in the format "Hey, you look like [Celebrity Name]!" If no match is found within the cutoff threshold, it labels the image as "Unknown".
