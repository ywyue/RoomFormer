import numpy as np
import os.path as osp

head = '''
<html>
<head>
<style>
td {text-align: center;}
</style>
</head>
<p>  
</p>
<br>
<table border="1">
'''

end = '''
</table>
<br>`
</html>
'''

def writeHTML(out_path, results_dirs):
    f = open(out_path, 'w')
    f.write(head + '\n')
    f.write('<tr>'
            '<td style="background-color:#FFFFFF"> ID  </td> '
            '<td style="background-color:#FFFFFF"> Input </td> '
            '<td style="background-color:#FFFFFF"> HAWP </td> '
            '<td style="background-color:#FFFFFF"> LETR </td> '
            '<td style="background-color:#FFFFFF"> HEAT (Ours) </td> '
            '<td style="background-color:#FFFFFF"> Ground-truth  </td> '
            '</tr>')

    wrong_s3d_annotations_list = [3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]
    file_ids = ['0{}'.format(x) for x in range(3250, 3500) if x not in wrong_s3d_annotations_list]
    permuted_ids = np.random.permutation(file_ids)
    file_ids = permuted_ids[:100]

    for file_id in file_ids:
        row_str = '<tr>'
        row_str += '<td> {} </td>'.format(file_id)
        for dir_idx, result_dir in enumerate(results_dirs):
            if dir_idx == 0:
                pred_filepath = osp.join(result_dir, 'scene_{}_alpha.png'.format(file_id))
                row_str += '<td> <img src="{}" width="180"> </td>'.format(pred_filepath)
            else:
                pred_filepath = osp.join(result_dir, '{}.png'.format(file_id))
                row_str += '<td> <img src="{}" width="180"> </td>'.format(pred_filepath)
        row_str += '</tr>'
        f.write(row_str + '\n')

    f.write(end + '\n')


if __name__ == '__main__':
    results_dirs = ['viz_density', 'viz_hawp', 'viz_letr', 'viz_heat_th5', 'viz_gt']

    writeHTML(out_path='./indoor_qual.html', results_dirs=results_dirs)
