import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow_addons.metrics import F1Score

class_names = np.array(['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])


class KerasLayer(hub.KerasLayer):
    pass


# 1. 加载模型
custom_objects = {
    'F1Score': F1Score,
    'Precision': Precision,
    'Recall': Recall
}
with tf.keras.utils.custom_object_scope(custom_objects):
    try:
        model = load_model(r'C:\Users\Lawaiian\PycharmProjects\dlmi\res2_10k_1_model.h5',
                           custom_objects={'KerasLayer': KerasLayer})
    except Exception as e:
        print(f"反序列化错误: {e}")
        import traceback

        traceback.print_exc()

# 2. 加载图像进行预处理
img_path = r'C:\Users\Lawaiian\PycharmProjects\dlmi\imgs\test.jpg'

# 3. 进行推断
img = image.load_img(img_path, target_size=(600, 450))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = model.predict(img_array)

# 4. 输出预测结果
print(predictions)
print(len(predictions[0]))

sorted_indices = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

# 提取前五个元素的索引
top_five_indices = [index for index, value in sorted_indices[:7]]

file = open(r"C:\Users\Lawaiian\IdeaProjects\dog-backend\cb-admin\src\main\java\com\test\cbadmin\upload\result.txt",
            'w')
print("前五个最大元素的索引:", top_five_indices)

# for idx in top_five_indices:
#     print(class_names[idx], predictions[0][idx])
#     res = str(class_names[idx]) + " " + str(predictions[0][idx])
#     file.write(res)
#     file.write('\n')
res = str(class_names[0]) + " " + str(predictions[0][0])
file.write(res)

file.close()
