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



-- Lua implementation of PHP scandir function
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a "'..directory..'"'):lines() do
        if filename:match "%.png$" then
            i = i + 1
            t[i] = filename
        end
    end
    return t, i
end

fileDir = '/Users/yusheng/data/fooling_img/fooling_5k/run_5'
dstPath = '/Users/yusheng/data/fooling_img/confident_fooling'
fileList, total_files = scandir( fileDir )

-- Setting up networks 

prototxt_name = '/Users/yusheng/data/caffe_models/caffenet-yos/caffenet-yos-deploy.prototxt'
binary_name = '/Users/yusheng/data/caffe_models/caffenet-yos/caffenet-yos-weights'





print '==> Loading network'

net = loadcaffe.load(prototxt_name, binary_name, 'cudnn')
--net.modules[#net.modules] = nil -- remove the top softmax






-- as we want to classify, let's disable dropouts by enabling evaluation mode
net:evaluate()

print '==> Loading synsets'
synset_words = load_synset()

print '==> Loading image and imagenet mean'
img_mean_name = 'ilsvrc_2012_mean.t7'
img_mean = torch.load(img_mean_name).img_mean:transpose(3,1)



local total_copies = 0
for i=1, total_files do
    im = image.load( fileDir..'/'..fileList[i] )

    -- print '==> Preprocessing'

    -- Have to resize and convert from RGB to BGR and subtract mean
    I = preprocess(im, img_mean)
    sorted_prob = net:forward(I:cuda()):view(-1):float():sort(true)
    print(sorted_prob[1])
    
    if sorted_prob[1] >= 0.9 then
        --copy the source file to the destination file
        local rfilePath = fileDir..'/'..fileList[i]
        local wfilePath = dstPath..'/'..fileList[i]

        local rfh = io.open( rfilePath, "rb" )
        local wfh = io.open( wfilePath, "wb" )
        local data = rfh:read( "*a" )
        if not ( data ) then
            print( "read error!" )
            return false
        else
            if not ( wfh:write( data ) ) then
                print( "write error!" )
                return false
            end
        end        
        
        --clean up file handles
        rfh:close()
        wfh:close()
        print( fileList[i]..'is copied !!!')
        
        total_copies = total_copies + 1
    end
end

print( 'Total cpoies : '.. total_copies )