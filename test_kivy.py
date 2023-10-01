# -*- coding: utf-8 -*-
# Contatct: AI-Lab - Smart Things
# Basic Kivy Testing

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

class DrawingWidget(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)
    
    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def clear_canvas(self):
        self.canvas.clear()

class InputWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_input = TextInput(pos=(10, 10), size=(300, 40), multiline=False)
        self.add_widget(self.text_input)

class DrawingApp(App):
    def build(self):
        root = Widget()
        drawing_widget = DrawingWidget()
        input_widget = InputWidget()
        clear_button = Button(text="Xóa", pos=(0, 0))
        clear_button.bind(on_release=drawing_widget.clear_canvas)
        root.add_widget(input_widget)
        root.add_widget(drawing_widget)
        root.add_widget(clear_button)
        return root

if __name__ == '__main__':
    DrawingApp().run()
