import gunpowder as gp
import numpy as np


class ZeroSource(gp.BatchProvider):
    # Purpose of this class: to be called for a ROI 
    # that we already know is empty instead of prediction using the model.
    # It should write zeros as parent vectors, 
    # which can then be written to zarr file using ZarrWrite 
    # (just subsitute for Prediction node into the pipeline)
    def __init__(
            self,
            array,
            num_channels=3,
            array_spec=None):
        self.array = array
        self.num_channels = num_channels
        self.array_spec = array_spec if array_spec else gp.ArraySpec()
    
    def setup(self):
        inf_spec = self.array_spec.copy()
        inf_spec.roi = gp.Roi(offset=gp.Coordinate((0,0,0,0)), shape=gp.Coordinate((None, None, None, None))) 
        self.provides(self.array, inf_spec)


    def provide(self, request):
        voxel_size = self.array_spec.voxel_size
        batch = gp.Batch()
        
        request_spec = request[self.array]
        dataset_roi = request_spec.roi // voxel_size
        
        spec = self.array_spec.copy()
        spec.roi = request_spec.roi

        batch.arrays[self.array] = gp.Array(
                np.zeros(
                    (self.num_channels,) + dataset_roi.get_shape(),
                    dtype=self.array_spec.dtype
                    ),
                spec)

        return batch
       

if __name__=="__main__":
    #test zero source
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    roi = gp.Roi(offset=gp.Coordinate((0,0,0,0)), shape=gp.Coordinate((10, 10, 10, 10)))
    pipeline = (
            ZeroSource(
                parent_vectors,
                gp.ArraySpec(voxel_size=gp.Coordinate((1,5,1,1)),
                    dtype=np.float32)
                ) # + 
            # gp.Crop(parent_vectors, roi)
            )
    with gp.build(pipeline):
        request = gp.BatchRequest()
        request.add(parent_vectors, roi.get_shape())
        b = pipeline.request_batch(request)
        data = b[parent_vectors].data
        print(data.shape)
        print(np.any(data))
        print(data)

