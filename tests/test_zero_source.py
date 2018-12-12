import gunpowder as gp
import numpy as np
from linajea.gunpowder import ZeroSource
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


if __name__=="__main__":
    #test zero source
    parent_vectors = gp.ArrayKey('PARENT_VECTORS')
    roi = gp.Roi(offset=gp.Coordinate((0,0,0,0)), shape=gp.Coordinate((10, 10, 10, 10)))
    pipeline = (
            ZeroSource(
                parent_vectors,
                array_spec=gp.ArraySpec(voxel_size=gp.Coordinate((1,5,1,1)),
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

