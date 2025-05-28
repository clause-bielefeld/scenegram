from os.path import abspath, dirname, join

FILE_LOCATION = dirname(abspath(__file__))


class Configuration():
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        max_len = max([len(k) for k in vars(self).keys()])
        lines = [
            f'{k.ljust(max_len)} : {v}'
            for k, v in (vars(self).items())
        ]
        return '\n'.join(lines)


class Config(Configuration):
    def __init__(self):
        self.mode = 'grid'
        self.split = 'dense10'
        assert self.split in ['full', 'dense', 'dense10']
        self.config_location = FILE_LOCATION
        self.data_path = abspath(join(self.config_location, f'./kilogram/dataset/{self.split}.json'))
        self.snd_path = abspath(join(self.config_location, f'kilogram/analysis/behavior/measures/measures_data/{self.split}_SND.json'))
        self.tangram_dir = abspath(join(self.config_location, './kilogram/dataset/tangrams-svg'))
        self.context_dir = abspath(join(self.config_location, './generated_scenes/'))
        self.data_out_dir = abspath(join(self.config_location, './generated_data/'))
        self.items_out_dir = abspath(join(self.config_location, './generated_items/'))
        self.samples_out_dir = abspath(join(self.config_location, './generated_samples/'))
        self.img_base_url = 'URL'
        
        self.svg_attrs = {
            'fill': 'darkgray',
            'stroke': 'white'
        }

        self.size = 1024
        self.context_size_ratio = 2
        self.wrapper_size_ratio = 1.1
        self.wrapper_alpha = 200
        self.wrapper_fill = 'black'
        self.wrapper_outline = 'white'
        self.wrapper_border_width = 0
