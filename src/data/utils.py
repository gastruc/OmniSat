def subset_dict_by_filename(files_to_subset, dictionary):
    return {file : dictionary[file] for file in files_to_subset}

def filter_labels_by_threshold(labels_dict, area_threshold = 0.07):
    """
    Parameters
    ----------
    labels_dict: dict, {filename1: [(label, area)],
                        filename2: [(label, area), (label, area)],
                        ...
                        filenameN: [(label, area), (label, area)]}
    area_threshold: float
    
    Returns
    -------
    filtered: dict, {filename1: [label],
                     filename2: [label, label],
                     ...
                     filenameN: [label, label]}
    """
    filtered = {}
    
    for img in labels_dict:
        for lbl, area in labels_dict[img]:
            # if area greater than threshold we keep the label
            if area > area_threshold:
                # init the list of labels for the image
                if img not in filtered:
                    filtered[img] = []
                # add only the label, since we won't use area information further
                filtered[img].append(lbl)
                
    return filtered