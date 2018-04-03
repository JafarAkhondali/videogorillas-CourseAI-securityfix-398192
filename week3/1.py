# cap_file = '../datasets/coco/annotations/captions_train2017.json'
# caps = COCO(cap_file)
#
# coco_files = COCO('../datasets/coco/annotations/instances_train2017.json')
# img_ids = coco_files.getImgIds()
# imgs = coco_files.loadImgs(img_ids)
#
# vocabulary = defaultdict(0)

# batch = []
# paths = []
# n = 0
#
# for ann_id in caps.anns:
#     ann = caps.anns[ann_id]
#     words = ann['caption'].replace(',', ' ').split(' ')
#     for w in words:
#         vocabulary[w] += 1
#
#     image_fn = imgs[ann['image_id']]['file_name']
#
#     img = imread(in_path + image_fn, mode='RGB')
#
#     paths.append(image_fn)
#     batch.append(img)
#     n += 1
#
#     if n == batch_size:
#         x = np.array(batch, dtype='float')
#         x = preprocess_input(x)
#
#         batch = []
#         n = 0
