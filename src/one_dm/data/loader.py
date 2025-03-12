import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F

text_path = {'train': 'IAM64_train.txt',
             'test': 'IAM64_test.txt'}

generate_type = {'iv_s':['train', 'data/in_vocab.subset.tro.37'],
                'iv_u':['test', 'data/in_vocab.subset.tro.37'],
                'oov_s':['train', 'data/oov.common_words'],
                'oov_u':['test', 'data/oov.common_words']}

# define the letters and the width of style image
letters = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
style_len = 352

"""prepare the IAM dataset for training"""
class IAMDataset(Dataset):
    def __init__(self, image_path, style_path, laplace_path, type, content_type='unifont', max_len=9):
        self.max_len = max_len
        self.style_len = style_len
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)
        self.laplace_path = os.path.join(laplace_path, type)
        self.data_dict = self.load_data(text_path[type])

        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        #self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        self.con_symbols = self.get_symbols(content_type)
        self.laplace = torch.tensor([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=torch.float
                                    ).to(torch.float32).view(1, 1, 3, 3).contiguous()



    def load_data(self, data_path):
        """加载数据集"""
        # 如果 data_path 是相对路径，则使用 image_path 的父目录作为基准
        if not os.path.isabs(data_path):
            data_path = os.path.join(os.path.dirname(os.path.dirname(self.image_path)), "data", data_path)
        
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                if len(transcription) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label':transcription}
                idx += 1
        return full_dict

    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2) # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]
        
        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])
        
        '''style images'''
        style_images = [style_image/255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image/255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images

    def get_symbols(self, input_type):
        """
        获取给定输入类型的符号集
        
        参数:
            input_type: 符号集类型，通常为'unifont'
        
        返回:
            contents: 包含所有字符符号的张量，形状为 [n_chars, h, w]
        """
        pickle_path = f"data/{input_type}.pickle"
        
        # 检查文件是否存在
        if not os.path.exists(pickle_path):
            print(f"警告: {pickle_path} 文件不存在，尝试自动创建")
            # 创建简单的模拟数据
            self._create_simple_mock_symbols(pickle_path)
                
        try:
            with open(pickle_path, "rb") as f:
                symbols = pickle.load(f)
                
            symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
            contents = []
            
            # 检查是否所有字符都在符号集中
            missing_chars = []
            for char in self.letters:
                if ord(char) in symbols:
                    symbol = torch.from_numpy(symbols[ord(char)]).float()
                    contents.append(symbol)
                else:
                    missing_chars.append(char)
                    # 为缺失字符创建空白符号
                    if contents:  # 确保已有至少一个符号来获取形状
                        symbol = torch.zeros_like(contents[0])
                    else:
                        symbol = torch.zeros((32, 32), dtype=torch.float32)
                    contents.append(symbol)
            
            # 如果有缺失字符，输出警告
            if missing_chars:
                print(f"警告: 在 {pickle_path} 中找不到以下字符: {''.join(missing_chars)}")
                
            # 添加PAD_TOKEN
            contents.append(torch.zeros_like(contents[0]))
            contents = torch.stack(contents)
            return contents
            
        except Exception as e:
            print(f"加载 {pickle_path} 时出错: {str(e)}，创建新的模拟数据")
            self._create_simple_mock_symbols(pickle_path)
            return self.get_symbols(input_type)  # 递归调用重试加载
        
    def _create_simple_mock_symbols(self, pickle_path):
        """创建简单的模拟符号数据"""
        # 确保data目录存在
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        
        # 创建模拟符号数据
        mock_data = []
        for char in self.letters:
            # 简单地为每个字符创建一个32x32的随机矩阵
            mock_symbol = {
                'idx': [ord(char)],
                'mat': np.random.rand(32, 32).astype(np.float32)
            }
            mock_data.append(mock_symbol)
        
        # 保存到文件
        with open(pickle_path, 'wb') as f:
            pickle.dump(mock_data, f)
        
        print(f"已创建简单模拟符号数据: {pickle_path}")
       
    def __len__(self):
        return len(self.indices)

    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        # 过滤掉不在 letter2index 中的字符
        filtered_labels = [i for i in labels if i in self.letter2index]
        if not filtered_labels:  # 如果过滤后为空，使用默认字符
            filtered_labels = ['_']  # 使用空格作为默认字符
        
        ll = [self.letter2index[i] for i in filtered_labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32) # [2, h , w] achor and positive
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32) # [2, h , w] achor and positive

        return {'img':image,
                'content':label, 
                'style':style_ref,
                "laplace":laplace_ref,
                'wid':int(wr_id),
                'transcr':transcr,
                'image_name':image_name}


    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])

            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])
            
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref # invert the image
        return {'img':imgs, 'style':style_ref, 'content':content_ref, 'wid':wid, 'laplace':laplace_ref,
                'target':target, 'target_lengths':target_lengths, 'image_name':image_name}


"""random sampling of style images during inference"""
class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, lapalce_path, ref_num) -> None:
        self.style_path = style_path
        self.laplace_path = lapalce_path
        self.author_id = os.listdir(os.path.join(self.style_path))
        self.style_len = style_len
        self.ref_num = ref_num
    
    def __len__(self):
        return self.ref_num
    
    def get_style_ref(self, wr_id): # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            laplace_image = cv2.imread(os.path.join(self.laplace_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image/255.0
        laplace_image = laplace_image/255.0
        return style_image, laplace_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref, laplace_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0)
            laplace_ref = laplace_ref.to(torch.float32)
            wid = idx
            batch.append({'style':style_ref, 'laplace':laplace_ref, 'wid':wid})
        
        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width], dtype=torch.float32)
        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)
        
        return {'style':style_ref, 'laplace':laplace_ref,'wid':wid_list}

"""prepare the content image during inference"""    
class ContentData(IAMDataset):
    def __init__(self, content_type='unifont') -> None:
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        
        # 检查unifont.pickle文件是否存在，如果不存在则自动创建
        if content_type == 'unifont':
            unifont_path = "data/unifont.pickle"
            if not os.path.exists(unifont_path):
                print(f"警告: {unifont_path} 文件不存在，自动创建模拟文件")
                self._create_mock_unifont_pickle(unifont_path)
        
        self.con_symbols = self.get_symbols(content_type)
        # 缓存已计算的内容
        self._content_cache = {}
    
    def _create_mock_unifont_pickle(self, unifont_path):
        """
        创建模拟的unifont.pickle文件
        
        这个方法会创建一个包含所有字符字形的模拟数据文件，用于测试和开发
        每个字符都会有一个独特的视觉表示，以便于区分
        
        参数:
            unifont_path: 保存模拟unifont文件的路径
        """
        # 确保data目录存在
        os.makedirs("data", exist_ok=True)
        
        # 创建模拟符号数据
        mock_data = []
        for char in self.letters:
            # 为每个字符创建一个32x32的随机矩阵作为其字体表示
            mock_symbol = {
                'idx': [ord(char)],
                'mat': np.zeros((32, 32), dtype=np.float32)
            }
            
            # 在矩阵中央绘制一个简单的表示
            h, w = mock_symbol['mat'].shape
            center_h, center_w = h // 2, w // 2
            size = 10
            
            # 根据字符的ASCII码值创建不同的图案
            ascii_val = ord(char)
            
            # 使用更多样化的模式来增加视觉区分度
            pattern_type = ascii_val % 8
            
            if pattern_type == 0:  # 方形
                mock_symbol['mat'][center_h-size//2:center_h+size//2, 
                                  center_w-size//2:center_w+size//2] = 1.0
            elif pattern_type == 1:  # 竖线
                mock_symbol['mat'][center_h-size:center_h+size, center_w-2:center_w+2] = 1.0
            elif pattern_type == 2:  # 横线
                mock_symbol['mat'][center_h-2:center_h+2, center_w-size:center_w+size] = 1.0
            elif pattern_type == 3:  # 十字
                mock_symbol['mat'][center_h-size:center_h+size, center_w-2:center_w+2] = 1.0
                mock_symbol['mat'][center_h-2:center_h+2, center_w-size:center_w+size] = 1.0
            elif pattern_type == 4:  # 圆形（近似）
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i - center_h) ** 2 + (j - center_w) ** 2)
                        if dist < size / 2:
                            mock_symbol['mat'][i, j] = 1.0
            elif pattern_type == 5:  # 对角线 \
                for i in range(-size//2, size//2):
                    if 0 <= center_h + i < h and 0 <= center_w + i < w:
                        mock_symbol['mat'][center_h + i, center_w + i] = 1.0
            elif pattern_type == 6:  # 对角线 /
                for i in range(-size//2, size//2):
                    if 0 <= center_h + i < h and 0 <= center_w - i < w:
                        mock_symbol['mat'][center_h + i, center_w - i] = 1.0
            else:  # 点阵
                step = size // 3
                for i in range(-size//2, size//2, step):
                    for j in range(-size//2, size//2, step):
                        if 0 <= center_h + i < h and 0 <= center_w + j < w:
                            mock_symbol['mat'][center_h + i, center_w + j] = 1.0
            
            # 在图像边缘添加字符的ASCII值，增加区分度
            ascii_str = str(ascii_val).zfill(3)
            for i, digit in enumerate(ascii_str):
                if i < 3 and i < w//8:
                    # 在图像顶部添加数字
                    d = int(digit)
                    for bit in range(4):
                        if d & (1 << bit):
                            r, c = 2, i * 8 + bit * 2
                            if 0 <= r < h and 0 <= c < w:
                                mock_symbol['mat'][r, c] = 1.0
            
            mock_data.append(mock_symbol)
        
        # 保存到文件
        os.makedirs(os.path.dirname(unifont_path), exist_ok=True)
        with open(unifont_path, 'wb') as f:
            pickle.dump(mock_data, f)
        
        print(f"已创建模拟 {unifont_path} 文件，包含 {len(mock_data)} 个字符")
       
    def get_content(self, label, device=None):
        """
        根据给定文本标签生成内容参考图像
        
        参数:
            label: 文本标签(字符串)或包含文本的列表/张量
            device: 输出张量的设备，默认为None(使用默认设备)
            
        返回:
            content_ref: 形状为 [batch_size, seq_len, h, w] 的张量，表示文本的视觉特征
        """
        # 检查缓存
        cache_key = str(label)
        if cache_key in self._content_cache:
            content = self._content_cache[cache_key]
            if device is not None:
                content = content.to(device)
            return content
        
        # 不同类型的输入处理
        if isinstance(label, str):
            # 从字符串生成内容
            word_arch = [self.letter2index[i] for i in label if i in self.letter2index]
            if not word_arch:  # 确保至少有一个字符
                word_arch = [self.letter2index['_']]  # 使用默认字符
                
            content_ref = self.con_symbols[word_arch]
            content_ref = 1.0 - content_ref
            content_ref = content_ref.unsqueeze(0)  # 添加批次维度
            
        elif isinstance(label, (list, tuple)) and all(isinstance(i, str) for i in label):
            # 处理字符串列表
            batch_contents = []
            for text in label:
                text_content = self.get_content(text)  # 递归调用单个字符串的处理
                batch_contents.append(text_content)
            content_ref = torch.cat(batch_contents, dim=0)
            
        elif isinstance(label, (list, tuple)) and all(isinstance(i, int) for i in label):
            # 处理整数列表（假设是字符索引）
            word_arch = [i for i in label if 0 <= i < len(self.letters)]
            if not word_arch:
                word_arch = [self.letter2index['_']]
                
            content_ref = self.con_symbols[word_arch]
            content_ref = 1.0 - content_ref
            content_ref = content_ref.unsqueeze(0)
            
        elif isinstance(label, torch.Tensor):
            # 处理张量（可能已经是内容表示或需要转换）
            if len(label.shape) == 2 and label.shape[1] == 1:
                # 假设是字符索引张量
                word_arch = label.cpu().numpy().flatten().tolist()
                valid_indices = [i for i in word_arch if 0 <= i < len(self.letters)]
                if not valid_indices:
                    valid_indices = [self.letter2index['_']]
                    
                content_ref = self.con_symbols[valid_indices]
                content_ref = 1.0 - content_ref
                content_ref = content_ref.unsqueeze(0)
                
            elif len(label.shape) >= 3:
                # 假设已经是内容表示
                content_ref = label
                
            else:
                # 未知张量格式，回退到默认处理
                print(f"警告: 无法识别的张量格式: {label.shape}，使用默认内容")
                content_ref = self.get_content("error")
        else:
            # 未知类型，使用默认内容
            print(f"警告: 无法识别的内容类型: {type(label)}，使用默认内容")
            content_ref = self.get_content("error")
        
        # 移动到指定设备
        if device is not None:
            content_ref = content_ref.to(device)
            
        # 缓存结果
        self._content_cache[cache_key] = content_ref.detach().clone()
            
        return content_ref
        
    def get_random_content(self, batch_size=1, length=None, device=None):
        """
        生成随机内容供测试使用
        
        参数:
            batch_size: 批次大小
            length: 每个样本的字符长度，如果为None则随机生成
            device: 指定输出张量的设备，如果为None则使用默认设备
            
        返回:
            contents: 形状为 [batch_size, seq_len, h, w] 的张量
        """
        if length is None:
            length = random.randint(3, 8)  # 随机长度的单词
            
        contents = []
        for _ in range(batch_size):
            # 随机选择字符
            random_chars = random.choices(list(self.letter2index.keys()), k=length)
            content = self.get_content(''.join(random_chars))
            contents.append(content)
            
        # 将内容合并为一个批次
        result = torch.cat(contents, dim=0) if batch_size > 1 else contents[0]
        
        # 如果指定了设备，移动到该设备
        if device is not None:
            result = result.to(device)
            
        return result