% segmentWound.m
function [mask,skinRing] = segmentWound(img,net)
% SEGMENTWOUND Segment a wound and return a surrounding skin-ring mask.
%   [MASK,SKINRING] = segmentWound(IMG,NET) predicts a binary wound mask
%   using a trained DeepLab v3+ dlnetwork NET. MASK and SKINRING have the
%   original image height and width. The ring is a dilation of MASK minus
%   MASK itself. NET must have been trained with three-channel RGB inputs.
%
%   IMG should be a grayscale or RGB image array, typically returned by
%   imread. Floating-point IMG values may be in [0,1] or [0,255].

if nargin~=2
    error("WoundWatch:InvalidInputCount","Usage: [mask,skinRing] = segmentWound(img,net).");
end
if isempty(img) || ~(isnumeric(img) || islogical(img)) || ~(ismatrix(img) || ndims(img)==3)
    error("WoundWatch:InvalidImage","img must be a nonempty grayscale or RGB numeric image.");
end
if ~isa(net,"dlnetwork")
    error("WoundWatch:InvalidNetwork","net must be a trained dlnetwork.");
end
if ndims(img)==3 && size(img,3)~=3 && size(img,3)~=4
    error("WoundWatch:InvalidImageChannels","img must have one, three, or four channels.");
end

originalSize=size(img,[1 2]);
if ndims(img)==2
    img=repmat(img,1,1,3);
elseif size(img,3)==4
    img=img(:,:,1:3);
end
if isfloat(img)
    if isempty(img) || any(~isfinite(img(:)))
        error("WoundWatch:InvalidImageValues","img must contain finite values.");
    end
    if max(img(:))<=1
        img=im2uint8(img);
    else
        img=uint8(min(max(round(double(img)),0),255));
    end
elseif ~isa(img,"uint8")
    img=im2uint8(img);
end

try
    networkInputSize=net.Layers(1).InputSize;
catch
    error("WoundWatch:UnknownNetworkInputSize", ...
        "Could not read input size from net.Layers(1).InputSize.");
end
if numel(networkInputSize)<3 || networkInputSize(3)~=3
    error("WoundWatch:InvalidNetworkInputSize","Network input must be H-by-W-by-3.");
end
resized=imresize(img,networkInputSize(1:2));
scores=gather(extractdata(predict(net,single(resized))));
scores=squeeze(scores);
if ndims(scores)~=3 || size(scores,3)~=2
    error("WoundWatch:UnexpectedSegmentationOutput", ...
        "Expected H-by-W-by-2 semantic scores from the network.");
end
networkMask=scores(:,:,2)>=scores(:,:,1);
mask=imresize(networkMask,originalSize,"nearest");
mask=logical(mask);

% Approximate 12 pixels at model-input resolution, scaled to original image size.
ringRadius=max(1,round(12*min(originalSize)/min(networkInputSize(1:2))));
dilated=imdilate(mask,strel("disk",ringRadius,0));
skinRing=dilated & ~mask;
end

