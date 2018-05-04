from PIL import Image
import numpy as np
import h5py
import json
import os

from IPython.display import display
import ipywidgets as widgets
from numbers import Number, Integral
from io import BytesIO

data_dir = "/unrshare/LESCROARTSHARE/IntroToEncodingModels/"

def img_array_to_png(img_array, target_size=None):
    img = Image.fromarray(img_array.astype('uint8'))
    if target_size is not None:
        img = img.resize(target_size)
    b = BytesIO()
    img.save(b, 'png')
    return b.getvalue()

def get_gray_image(shape=(96, 96)):
    return np.ones(shape + (3,), dtype='uint8') * np.uint8(128)

gray_png = img_array_to_png(get_gray_image())

class LHImages:
    def __init__(self):
        self.images_file = os.path.join(data_dir, "color_natims_images.hdf")
        self.hdf_file = h5py.File(self.images_file)
        self.train_len = self.hdf_file['est'].shape[-1]
        self.test_len = self.hdf_file['val'].shape[-1]
    
    def __len__(self):
        return self.train_len + self.test_len
    
    def __getitem__(self, item):
        if not isinstance(item, Number):
            raise NotImplementedError("Only one index at a time currently")
        if not isinstance(item, Integral):
            raise ValueError("Index must be of integer type")
        
        if item < self.test_len:
            index = item
            array = 'val'
        else:
            index = item - self.test_len
            array = 'est'
        return self.hdf_file[array][..., index]

class ClickResponsiveToggleButtons(widgets.ToggleButtons):
    """Added in from https://github.com/jupyter-widgets/ipywidgets/issues/763"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_handlers = widgets.CallbackDispatcher()
        self.on_msg(self._handle_button_msg)
        pass

    def on_click(self, callback, remove=False):
        """Register a callback to execute when the button is clicked.
        The callback will be called with one argument, the clicked button
        widget instance.
        Parameters
        ----------
        remove: bool (optional)
            Set to true to remove the callback from the list of callbacks.
        """
        self._click_handlers.register_callback(callback, remove=remove)

    def _handle_button_msg(self, _, content, buffers):
        """Handle a msg from the front-end.
        Parameters
        ----------
        content: dict
            Content of the msg.
        """
        if content.get('event', '') == 'click':
            self._click_handlers(self)
            



class ImageTagger:
    
    def __init__(self, tag_specs, tag_range=None, tag_filename=None, my_feature=None, ok=None):
        
        self.tag_specs = tag_specs
        self.tag_range = tag_range
        self.tag_filename = tag_filename
        self.my_feature = my_feature
        self.images = LHImages()
        self.ok = ok
        
        self.init_range()
        self.load_or_init_tags(self.tag_filename)
        self.create_widgets()
        self.set_callbacks()
        self.render()

    def init_range(self):
        if self.tag_range is None:
            self.tag_indices = np.array(sorted(list(range(len(self.images)))))
        else:
            self.tag_indices = np.array(sorted(list(self.tag_range)))
        
        self.id_to_img_index = dict(enumerate(self.tag_indices))
        self.img_index_to_id = {v:k for k, v in self.id_to_img_index.items()} # flip index around
    
    def next_img(self, cur_img):
        img_id = self.img_index_to_id.get(cur_img, None)
        if img_id is None:
            all_higher = self.tag_indices[self.tag_indices > cur_img]
            next_higher = all_higher.min() if len(all_higher) else None
            if next_higher is None:
                return None
            img_id = self.img_index_to_id[next_higher] - 1  # -1 because otherwise it does two steps
        return self.id_to_img_index.get(img_id + 1, None)
    
    def prev_img(self, cur_img):
        img_id = self.img_index_to_id.get(cur_img, None)
        if img_id is None:
            all_lower = self.tag_indices[self.tag_indices < cur_img]
            next_lower = all_lower.max() if len(all_lower) else None
            if next_lower is None:
                return None
            img_id = self.img_index_to_id[next_lower] + 1 # + 1 because otherwise it does two steps
        return self.id_to_img_index.get(img_id - 1, None)
    
    def load_or_init_tags(self, tag_names=None, tag_filename=None):
        if tag_names is None:
            tag_names = list(self.tag_specs.keys())
        if tag_filename is None:
            tag_filename = self.tag_filename or os.path.expanduser("~/LH_tags.json")
            
        if os.path.exists(tag_filename):
            tag_file_content = json.load(open(tag_filename, 'r'))
            self.all_tags = tag_file_content['tag_specs']
        else:
            # If file doesn't exist yet, init
            all_tags = dict()
            for tag_name in self.tag_specs.keys():
                all_tags[tag_name] = ['untagged'] * len(self.images)
            self.all_tags = all_tags
    
    def save_tags(self, tag_filename=None):
        if tag_filename is None:
            tag_filename = self.tag_filename or os.path.expanduser("~/LH_tags.json")
    
        tag_file_content = dict(tag_specs=self.all_tags)
        # Add user email:
        if self.ok is not None:
            tag_file_content['user'] = self.ok.assignment.get_student_email()
        with open(tag_filename, 'w') as f:
            json.dump(tag_file_content, f)
            
    def get_start_value(self):
        starting_value = min(self.tag_range) if self.tag_range is not None else 0
        return starting_value
    
    def create_widgets(self):
        starting_value = self.get_start_value()
        self.img = widgets.Image(value=gray_png, height=512, width=512)

        self.slider = widgets.IntSlider(value=starting_value, min=0, max=len(self.images) - 1, step=1)
        self.b_back = widgets.Button(icon="angle-left")
        self.b_forward = widgets.Button(icon="angle-right")
        
        self.tagging_widgets = {tag_name: ClickResponsiveToggleButtons(options=tag_options, description=tag_name)
                                            if isinstance(tag_options, list)
                                               else tag_options
                               for tag_name, tag_options in self.tag_specs.items()}

        self.feature_taggers_box = widgets.VBox()
        self.feature_type_selector = widgets.SelectMultiple(options=sorted(self.tag_specs.keys()))

        self.b_previous_untagged = widgets.Button(icon="angle-double-left")
        self.b_next_untagged = widgets.Button(icon="angle-double-right")
        
        self.progress_img = widgets.Image(value=img_array_to_png(get_gray_image((50, 200))), height=50)
        self.b_save = widgets.Button(description='save')
        
        self.progress_bar = widgets.IntProgress(min=0, max=len(self.tag_indices), value=0)
        self.remaining_label = widgets.Label(value="")
        
        
    def get_progress_img(self, num_pixel_rows_per_tag=100):
        active_tags = self.feature_type_selector.value
        tagged = (np.stack([self.all_tags[at] for at in active_tags], axis=0) != 'untagged').astype('int')
        taggable = np.zeros_like(tagged)
        taggable[:, self.tag_indices] = tagged[:, self.tag_indices] + 1
        colors = np.array([[128, 128, 128], [255, 0, 0], [0, 255, 0]]).astype('uint8')[taggable]
        colors[:, self.slider.value] = 255, 255, 0
        output_shape = (colors.shape[0] * num_pixel_rows_per_tag,) + colors.shape[1:]
        output = np.empty(output_shape, dtype='uint8')
        output.reshape(colors.shape[0], num_pixel_rows_per_tag, -1)[:] = colors.reshape(colors.shape[0], 1, -1)
        return img_array_to_png(colors)

    def update_progress_bar(self):
        active_tags = self.feature_type_selector.value
        tagged = (np.stack([self.all_tags[at] for at in active_tags], axis=0) != 'untagged')
        num_tagged = tagged[:, self.tag_indices].sum()
        self.progress_bar.value = num_tagged
        
        remaining = len(self.tag_indices) * tagged.shape[0] - num_tagged
        self.remaining_label.value = "{remaining} remaining".format(remaining=remaining) if remaining else "DONE"
        

    
    def update_progress(self):
        self.progress_img.value = self.get_progress_img()
        self.update_progress_bar()
    

    def update_img(self, new_index=None):
        if new_index is None:
            new_index = self.slider.value
        if new_index < 0:
            self.img.value = gray_png
        elif new_index >= len(self.images):
            self.img.value = gray_png
        else:
            self.img.value = img_array_to_png(self.images[new_index])

    def update_tag_widgets(self, new_index=None):
        if new_index is None:
            new_index = self.slider.value

        for tag_name, tag_widget in self.tagging_widgets.items():
            tag_widget.value = self.all_tags[tag_name][new_index]

    
    def get_previous_untagged(self):
        current_value = self.slider.value
        all_previous = [[i for i, tag_value in enumerate(self.all_tags[tag_name][:current_value])
                            if tag_value == 'untagged' and i in self.tag_indices]
                        for tag_name in self.feature_type_selector.value]
        maxima = [max(l) for l in all_previous if l]
        previous = max(maxima) if maxima else None
        return previous

    
    def get_next_untagged(self):
        current_value = self.slider.value
        all_next = [[i for i, tag_value in enumerate(self.all_tags[tag_name])
                            if i > current_value and tag_value == 'untagged' and i in self.tag_indices]
                        for tag_name in self.feature_type_selector.value]
        minima = [min(l) for l in all_next if l]
        the_next = min(minima) if minima else None
        return the_next
    
    
    def goto_next(self, *args):
        """Go to next image on tagging event. This function exists in order to be able to decide what to do"""
        next_untagged = self.get_next_untagged()
        if next_untagged is not None:
            self.slider.value = next_untagged
        else:
            self.update_progress()


    def set_callbacks(self):
        
        # slider buttons
        def decrement_slider(*args):
            prev = self.prev_img(self.slider.value)
            if prev is None: # need to do it this way because 0 evaluates to False ...
                prev = self.slider.value
            self.slider.value = prev
        def increment_slider(*args):
            self.slider.value = self.next_img(self.slider.value) or self.slider.value
        
        self.b_back.on_click(decrement_slider)
        self.b_forward.on_click(increment_slider)

        # Whenever slider value changes, update image and the tag widgets
        def slider_update(change):
            if change['name'] == 'value':
                new_index = change['new']
                # Change images
                self.update_img(new_index)
                # Change Tag content
                self.update_tag_widgets(new_index)
                # Change Progress Bar?
                self.update_progress()
        self.slider.observe(slider_update, 'value')
        
        # Next/previous untagged buttons
        def goto_next_untagged(*args):
            self.slider.value = self.get_next_untagged() or self.slider.value
        def goto_previous_untagged(*args):
            prev = self.get_previous_untagged()
            if prev is None: # need to do it this way because 0 evaluates to False ...
                prev = self.slider.value
            self.slider.value =  prev
        self.b_next_untagged.on_click(goto_next_untagged)
        self.b_previous_untagged.on_click(goto_previous_untagged)
        
            
        def get_tag_value_updater(tag_name):
            def tag_value_updater(tag_change):
                new_tag = tag_change['new']
                current_index = self.slider.value
                self.all_tags[tag_name][current_index] = new_tag
                
            return tag_value_updater
        
        for tag_name, tag_widget in self.tagging_widgets.items():
            tag_widget.observe(get_tag_value_updater(tag_name), 'value')
        
        # make only selected tag widgets visible
        def feature_type_selector_update(change):
            new_values = change['new']
            self.feature_taggers_box.children = [self.tagging_widgets[feature] for feature in new_values]
            for tag_name, tag_widget in self.tagging_widgets.items():
                tag_widget.on_click(self.goto_next, remove=True)
            self.tagging_widgets[new_values[-1]].on_click(self.goto_next)
            self.progress_bar.max = len(self.tag_indices) * len(new_values)
            self.update_progress()

        self.feature_type_selector.observe(feature_type_selector_update, 'value')
        
        # save button
        self.b_save.on_click(lambda x: self.save_tags())
        
        
    def render(self):
        # invoke feature_type_selector callback
        feature_type = self.my_feature
        # select used feature
        if feature_type is None:
            non_untagged_features = sorted([k for k, v in self.all_tags.items() if (np.unique(v) != 'untagged').any()])
            num_tags = [(np.array(self.all_tags[feature]) != 'untagged').sum() for feature in non_untagged_features]
            if num_tags != []:
                feature_type = non_untagged_features[np.argmax(num_tags)]
            else:
                feature_type = sorted(self.tagging_widgets.keys())[0]
            self.feature_type_selector.value = feature_type,
        else:
            if isinstance(feature_type, str):
                feature_type = feature_type,
            self.feature_type_selector.value = feature_type

        # invoke slider callback
        self.slider.value += 1
        self.slider.value -= 1
        
        
        self.slider_and_buttons = widgets.HBox([self.b_back, self.b_forward, self.slider])
        self.next_untagged_buttons_and_progress = widgets.HBox([self.b_previous_untagged,
                                                                self.b_next_untagged,
                                                                self.progress_bar,
                                                               self.remaining_label])
        
        self.feature_type_selector_accordion = widgets.Accordion(children=[self.feature_type_selector])
        self.feature_type_selector_accordion.set_title(0, "Feature")
        
        self.progress_accordion = widgets.Accordion(children=[widgets.HBox([widgets.Label(value="Progress"),
                                                             self.progress_img])])
        self.progress_accordion.set_title(0, "Tagging Status")
        self.progress_accordion.selected_index = None
        
        self.the_whole_thing = widgets.VBox([self.feature_type_selector_accordion,
                                             self.progress_accordion,
                                             self.img,
                                             self.feature_taggers_box,
                                             self.next_untagged_buttons_and_progress,
                                             self.slider_and_buttons,
                                             self.b_save])
    def _ipython_display_(self):
        display(self.the_whole_thing) if hasattr(self, 'the_whole_thing') else ""
