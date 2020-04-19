This repository is a collection of scripts for practicing Machine Learning. Each directory is a separate project.

## face-recognition-1

Follows instruction at [A Simple Introduction to Facial Recognition](https://www.analyticsvidhya.com/blog/2018/08/a-simple-introduction-to-facial-recognition-with-python-codes), contains code for a building a straightforward face recognition system using the `face_recognition` library.

Installation:

- [`dlib`](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
- [`face_recognition`](https://github.com/ageitgey/face_recognition#installation-options)

Usage:

```shell
$ cd face-recognition-1

$ python3 face-recognition-1.py my_image_1.jpg
Not matched: warren_buffett.jpeg
Not matched: mark_zuckerberg.jpg
Not matched: barack_obama.jpg
Not matched: jeff_bezos.jpg
Not matched: bill_gates.jpg

$ python3 face-recognition-1.py my_image_2.jpg
Not matched: warren_buffett.jpeg
Not matched: mark_zuckerberg.jpg
Not matched: barack_obama.jpg
Not matched: jeff_bezos.jpg
Matched: bill_gates.jpg

$ python3 face-recognition-1.py my_image_3.jpg
Not matched: warren_buffett.jpeg
Matched: mark_zuckerberg.jpg
Not matched: barack_obama.jpg
Not matched: jeff_bezos.jpg
Not matched: bill_gates.jpg
```

