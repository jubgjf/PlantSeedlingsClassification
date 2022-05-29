# Plant Seedlings Classification

|模型           |数据增强                       |优化器 |MicroF1Score   |MacroF1Score   |KaggleScore|  
|:-             |:-                             |:-     |:-             |:-             |:-         |
|VGG16          |无                             |Adam   |               |               |           |
|VGG16          |中心裁剪+色彩增强+水平翻转+旋转|SGD    |               |               |           |
|VGG16          |中心裁剪+色彩增强+水平翻转+旋转|Adam   |               |               |           |
|ResNet18       |无                             |Adam   |               |               |           |
|ResNet18       |中心裁剪+色彩增强+水平翻转+旋转|SGD    |               |               |           |
|ResNet18       |中心裁剪+色彩增强+水平翻转+旋转|Adam   |               |               |           |
|SENet18        |无                             |Adam   |               |               |           |
|SENet18        |中心裁剪+色彩增强+水平翻转+旋转|SGD    |               |               |           |
|SENet18        |中心裁剪+色彩增强+水平翻转+旋转|Adam   |               |               |           |
|SwinTransformer|预训练的feature_extractor      |SGD    |               |               |           |
|SwinTransformer|预训练的feature_extractor      |AdamW  |               |               |           |
