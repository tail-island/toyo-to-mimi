import numpy   as np
import os.path as path

from funcy        import count, juxt, map
from tkinter      import *
from tkinter.ttk  import *


class UselessAbsolutePitchFrame(Frame):
    def __init__(self, character_paths, master=None):
        super().__init__(master)

        self.character_images = tuple(map(lambda character_path: PhotoImage(file=path.join(character_path, 'image.ppm')), character_paths))
        self.character_small_images = tuple(map(lambda character_path: PhotoImage(file=path.join(character_path, 'small_image.ppm')), character_paths))

        self.master.title('ダメ絶対（？）音感')
        self.create_widgets()

        self.pack()

    def create_widgets(self):
        frame_1 = Frame(self)

        self.wave_canvas = Canvas(frame_1, width=256, height=256)
        self.wave_canvas.grid(row=0, column=0)

        self.character_canvas = Canvas(frame_1, width=256, height=256)
        self.character_canvas.grid(row=0, column=1)

        frame_1.pack()

        frame_2 = Frame(self)

        self.characters_canvas = Canvas(frame_2, width=512, height=64)
        self.characters_canvas.pack()

        frame_2.pack()

    def draw_predict_result(self, wave, character_indice):
        self.draw_wave(wave)
        self.draw_predicted_character(character_indice[0])
        self.draw_predicted_characters(character_indice)

        self.update()

    def draw_wave(self, wave):
        min_ys, max_ys = zip(*map(juxt(np.min, np.max), np.array_split(wave * 128 + 128, 256)))

        for object_id in self.wave_canvas.find_all():
            self.wave_canvas.delete(object_id)

        for x, min_y, max_y in zip(count(), min_ys, max_ys):
            self.wave_canvas.create_line(x, min_y, x, max_y)

    def draw_predicted_character(self, character_index):
        for object_id in self.character_canvas.find_all():
            self.character_canvas.delete(object_id)

        self.character_canvas.create_image(128, 128, image=self.character_images[character_index])

    def draw_predicted_characters(self, character_indice):
        for object_id in self.characters_canvas.find_all():
            self.characters_canvas.delete(object_id)

        for i in range(8):
            self.characters_canvas.create_image(i * 64 + 32, 32, image=self.character_small_images[character_indice[i]])
