

int prroi_pooling_forward_cuda(THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, int pooled_height, int pooled_width, float spatial_scale);

int prroi_pooling_backward_cuda(
    THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, THCudaTensor *output_diff, THCudaTensor *features_diff,
    int pooled_height, int pooled_width, float spatial_scale
);

int prroi_pooling_coor_backward_cuda(
    THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, THCudaTensor *output_diff, THCudaTensor *features_diff,
    int pooled_height, int pooled_width, float spatial_scal
);

