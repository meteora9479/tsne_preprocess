require 'loadcaffe'
require 'image'





-- Loads the mapping from net outputs to human readable labels
function load_synset()
  local file = io.open 'synset_words.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line,11))
  end
  return list
end






-- Converts an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')*255
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  -- subtract imagenet mean
  return im4 - image.scale(img_mean, 224, 224, 'bilinear')
end






-- Setting up networks 

prototxt_name = './caffe_models/caffenet-yos/caffenet-yos-deploy.prototxt'
binary_name = './caffe_models/caffenet-yos/caffenet-yos-weights'





print '==> Loading network'

net = loadcaffe.load(prototxt_name, binary_name, 'cudnn')
--net.modules[#net.modules] = nil -- remove the top softmax






-- as we want to classify, let's disable dropouts by enabling evaluation mode
net:evaluate()

print '==> Loading synsets'
synset_words = load_synset()

print '==> Loading image and imagenet mean'
image_name = './input_images/ILSVRC2012_val_00006451.jpg'
img_mean_name = 'ilsvrc_2012_mean.t7'

im = image.load(image_name)
img_mean = torch.load(img_mean_name).img_mean:transpose(3,1)

print '==> Preprocessing'




-- Have to resize and convert from RGB to BGR and subtract mean
I = preprocess(im, img_mean)






-- Propagate through the network and sort outputs in decreasing order and show 5 best classes
_,classes = net:forward(I:cuda()):view(-1):float():sort(true)
sorted_prob = net:forward(I:cuda()):view(-1):float():sort(true)

for i=1,5 do
  print('predicted class '..tostring(i)..': ', string.format('%f', sorted_prob[i]), synset_words[classes[i] ] )
end
