import torch
import numpy as np
import cv2

def compute_localized_depth(points:torch.Tensor, segmentation_mask:torch.Tensor, 
                            confidence_map:torch.Tensor, depth_map:torch.Tensor, window_size:int):
    """
    Extract depth information for the given points, by doing a weighted sum between confidence and depth maps over a sqaured window with size `window_size`.
    Also, data are masked by the provided `segmentation_mask`.
    Confidence map must be normalized --> values in [0,1] range.

    ------

    Returns:
    A tensor with shape (N, 4), where N was the number of points provided in input:
        - results[:, :2] are the coordinates (x,y) of the input points (the same in input)
        - results[:, 2] contain confidence score for each of the results, based on the confidence map. When this value is low, you should not trust the resulting information.
        - results[:, 3] are the estimated distances.

    """
    with torch.no_grad():
        # note that these maps are expressed in [y,x], while points are in [x,y]

        assert confidence_map.device == depth_map.device == segmentation_mask.device
        device = confidence_map.device
        points = points.to(torch.int16).unsqueeze(-1)

        n_points = len(points)
        (h, w) = segmentation_mask.shape
        half_window = window_size//2

        masks = segmentation_mask.expand(n_points, h, w)
        depth_map = depth_map.unsqueeze(0).expand(n_points, -1, -1)
        confidence_map = confidence_map.unsqueeze(0).expand(n_points, -1, -1)

        xs = torch.arange(w, device=device)[None, :]
        ys = torch.arange(h, device=device)[None, :]
        interest_areas_x = (xs >= (points[:,0]-half_window)) * (xs < (points[:,0]+half_window))
        interest_areas_x = interest_areas_x.unsqueeze(1).expand(-1, h, -1)

        interest_areas_y = (ys >= (points[:,1]-half_window)) * (ys < (points[:,1]+half_window))
        interest_areas_y = interest_areas_y.unsqueeze(2).expand(-1, -1, w)

        masks = masks * interest_areas_x * interest_areas_y

        # now, using the given masks, let-s extract a weighted sum of depths, and a confidence score

        confidence_map = confidence_map * masks
        depth_map = depth_map * masks

        conf_scores = confidence_map.sum(dim=(1,2)) / masks.count_nonzero((1,2))
        avg_dist = (depth_map * confidence_map).sum((1,2)) / confidence_map.sum((1,2))

        conf_scores[torch.isnan(conf_scores)] = 0
        avg_dist[torch.isnan(avg_dist)] = 0

        results = torch.empty((n_points, 4), dtype=torch.float16, device=device)
        results[:, :2] = points[:, :2].squeeze(-1)
        results[:, 2] = conf_scores
        results[:, 3] = avg_dist
        return results
    

def extract_depth_points(grid_size:tuple, segmentation_mask:torch.Tensor, confidence_map:torch.Tensor, depth_map:torch.Tensor, debug=False):
    """
    Using the provided depth map and the relative confidence, maps are subdivided in smaller windows, based on the provided `grid_size`. For each of these windows, it returns the coords, confidence and distance value of the points with best confidence.
    Also, data are masked by the provided `segmentation_mask`.

    ------

    Returns:
    A tensor with shape (N, 4), where `N = grid_size[0] * grid_size[1]`
        - results[:, :2] are the coordinates (x,y) of the input points (the same in input)
        - results[:, 2] contain confidence score for each of the results, based on the confidence map. When this value is low, you should not trust the resulting information.
        - results[:, 3] are the estimated distances.

    """

    with torch.no_grad():
        # note that these maps are expressed in [y,x], while points are in [x,y]

        assert confidence_map.device == depth_map.device == segmentation_mask.device
        assert segmentation_mask.shape == confidence_map.shape == depth_map.shape

        items_h, items_w = grid_size[0], grid_size[1]
        assert items_h > 0 and items_h <= confidence_map.shape[0]
        assert items_w > 0 and items_w <= confidence_map.shape[1]
                    

        device = confidence_map.device
        h,w = confidence_map.shape

        # let's mask conficence using segmentation
        confidence_map = confidence_map * segmentation_mask

        h_pixels = h//items_h
        w_pixels = w//items_w

        # in case of not perfect matching, let's avoid to process the last rows / columns
        h = h_pixels * items_h
        w = w_pixels * items_w
        confidence_map = confidence_map[:h, :w]

        

        x_dim = torch.arange(0,w,device=device).unsqueeze(0).expand_as(confidence_map)
        y_dim = torch.arange(0,h,device=device).unsqueeze(1).expand_as(confidence_map)
        
        conf = confidence_map.unfold(1, w_pixels, w_pixels).unfold(0, h_pixels, h_pixels).reshape(-1, h_pixels * w_pixels)
        x_dim = x_dim.unfold(1, w_pixels, w_pixels).unfold(0, h_pixels, h_pixels).reshape(-1, h_pixels * w_pixels)
        y_dim = y_dim.unfold(1, w_pixels, w_pixels).unfold(0, h_pixels, h_pixels).reshape(-1, h_pixels * w_pixels)
        
        best_confs, best_idx = torch.max(conf, dim=1, keepdim=False)

        results = torch.empty((items_h * items_w, 4), dtype=torch.float16, device=device)
        x = x_dim[range(x_dim.shape[0]) ,best_idx]
        y = y_dim[range(y_dim.shape[0]), best_idx]

        results[:, 0] = x
        results[:, 1] = y
        results[:, 2] = best_confs

        results[:, 3] = depth_map[y, x]

        if debug:
            depth = (depth_map * 255 / 9.0).to(torch.uint8)
            cv2.imshow("mask", (segmentation_mask.to(torch.uint8) * 255).cpu().numpy())
            cv2.imshow("confidence", confidence_map.cpu().numpy())
            cv2.imshow("depth", depth.cpu().numpy())
            base = confidence_map.cpu().numpy()
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
            cv2.imshow("masked confidence", base)
            for i in range(1, w//w_pixels):
                x = i * w_pixels
                a, b = (x, 0), (x, h)
                cv2.line(base, a, b, (0,0,255))
            
            for i in range(1, h//h_pixels):
                y = i * h_pixels
                a, b = (0, y), (w, y)
                cv2.line(base, a, b, (0,0,255))

            cv2.imshow("grid", base)

            maxes = base.copy()
            
            for i in range(results.shape[0]):
                p = results[i]
                cv2.circle(maxes, (int(p[0]), int(p[1])),radius=5, color=(0,255,0), thickness=2)

                if p[2] > 0.75:
                    cv2.circle(base, (int(p[0]), int(p[1])),radius=5, color=(0,255,0), thickness=2)
            
            cv2.imshow("maxes", maxes)
            cv2.imshow("final", base)
            cv2.waitKey()

        return results


def retrieve_distance(centers:torch.Tensor, distance_points:torch.Tensor):
    """
    centers is a tensor shaped [N, 2], where the inner dimension is formatted as [x, y]
    distance_points is a tensor shaped [M, 3], where the inner dimension is formatted as [x, y, distance]

    Returns:
        A tensor with shape (N,) with the distances of the corresp. points.
    """

    assert centers is not None
    assert centers.ndim == 2 and centers.shape[-1] == 2
    assert distance_points is not None
    assert distance_points.ndim == 2 and distance_points.shape[-1] == 3

    # This algorithm perform retrieving by searching for the 4 closest points to (x,y),
    # and doing an interpolation over them.
    centers = centers.unsqueeze(-1)
    distance_points = distance_points.to(torch.float32)
    centers = centers.to(torch.float32)
    planar_dist = torch.pow(distance_points[:, 0] - centers[:, 0], 2) + torch.pow(distance_points[:, 1] - centers[:, 1], 2)


    # Getting the k nearest points
    knn_indices = planar_dist.topk(4, largest=False, sorted=True)[1]

    closest_points = distance_points[knn_indices]
    planar_dist = (torch.pow(closest_points[:, :, 0] - centers[:, 0], 2) + torch.pow(closest_points[:, :, 1] - centers[:, 1], 2)).pow(0.5)
    similarity = 1 / planar_dist

    dist = (closest_points[:, :, -1] * similarity).sum(dim=-1)
    dist = dist / similarity.sum(dim=1)

    # since it could happen to get a distance zero, similarity in those cases will be nan values.
    # In these cases, out interest point is exactly one of the samples point, and we can just use the associated distance value.
    nan_indexes = torch.isnan(dist)
    dist[nan_indexes] = closest_points[nan_indexes][:, 0, -1]

    return dist




if __name__ == "__main__":

    #sample_points = [[0,0, 4], [1,1, 5], [0,1, 6], [1,0, 7], [3,3, 2]]
    #centers = [[2.9, 2.9], [0, 1]]
    #res = retrieve_distance(torch.tensor(centers), torch.tensor(sample_points))
    #print("results: ", res.shape, res)

    conf_map = torch.arange(6*4).view(6,4) /(6*4)
    dist = torch.randn_like(conf_map)
    segm = torch.ones_like(conf_map) - torch.cat((torch.eye(4), torch.zeros(4).unsqueeze(0), torch.zeros(4).unsqueeze(0)),dim=0)
    
    grid=(3,2)
    res = extract_depth_points(grid, segm, conf_map, dist)
    print(res)