import unittest
import numpy as np
from PIL import Image
from woundwatch.color_features import rgb_to_cielab
from woundwatch.visualization import colour_heatmap,compare_features

class ColourTests(unittest.TestCase):
    def test_cielab_reference(self):
        self.assertTrue(np.allclose(rgb_to_cielab(Image.new('RGB',(1,1),'white'))[0,0],[100,0,0],atol=.01))
        self.assertTrue(np.allclose(rgb_to_cielab(Image.new('RGB',(1,1),'black'))[0,0],[0,0,0],atol=.01))
        red=rgb_to_cielab(Image.new('RGB',(1,1),'red'))[0,0]
        self.assertTrue(np.allclose(red,[53.2408,80.0925,67.2032],atol=.01))

    def test_fixed_map_and_method_guard(self):
        a=colour_heatmap(Image.new('RGB',(20,20),(160,90,80)))
        b=colour_heatmap(Image.new('RGB',(40,40),(160,90,80)))
        self.assertTrue(np.array_equal(np.asarray(a)[0,0],np.asarray(b)[0,0]))
        drift,error=compare_features({'features':{'a':2},'colour_method':'v2'},{'features':{'a':1}})
        self.assertIsNone(drift);self.assertIn('differ',error)
        self.assertEqual(compare_features({'features':{'a':2}},{'features':{'a':1}}),({'a':1},None))
