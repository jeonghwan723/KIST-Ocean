import numpy as np
import torch

def weighted_l1_loss(fcst, gt, weight):

    '''
    fcst: Forecast
    gt: Groundtruth
    weight: weight values for loss
    '''

    loss = torch.mean(weight[:,None] * torch.abs(fcst - gt))

    return loss

def weighted_l2_loss(fcst, gt, weight):

    '''
    fcst: Forecast
    gt: Groundtruth
    weight: weight values for loss
    '''

    loss = torch.mean(weight[:,None] * torch.square(fcst - gt))

    return loss


def lat_weight(ydim, lat1=-90, lat2=90):

    # compute interval
    intv = (lat2 - lat1 + 1) / ydim

    # make latitude list
    lat = np.arange(lat1, lat2+1e-6, intv)

    # convert angular to radians
    lat = np.radians(lat)

    # compute weights
    weight = np.zeros((ydim))
    for i in range(ydim):
        weight[i] = np.cos(lat[i]) / np.sum(np.cos(lat)) * ydim

    return torch.as_tensor(weight)

