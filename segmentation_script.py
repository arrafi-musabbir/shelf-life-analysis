import cv2
import shutil
import json
import os
import pandas as pd
from roboflow import Roboflow
from shapely.geometry import Polygon
import time 
import json

def processAndsegmentImages(day):
    if os.path.exists(os.path.join(os.getcwd(), 'segmentation_results')):
        seg_results_dir = os.path.join(os.getcwd(), 'segmentation_results')
    else:
        os.mkdir(os.path.join(os.getcwd(), 'segmentation_results'))
        seg_results_dir = os.path.join(os.getcwd(), 'segmentation_results')
        
    root_dir = os.path.join(seg_results_dir,day+'_segmentation_results')
    try:
        try:
            shutil.rmtree(root_dir)
            os.mkdir(root_dir)
            print('created root directory: {}'.format(root_dir))
        except:
            os.mkdir(root_dir)
            print('created root directory: {}'.format(root_dir))
    except:
        print('cannot create root directory: {}'.format(root_dir))
        
    images_dir = os.path.join(root_dir,day+'_images')
    df = pd.read_excel("readings/readings_"+day+".xlsx")
    orientation = ['angle', 'front', 'top']
    count = 0
    try:
        try:
            shutil.rmtree(images_dir)
            os.mkdir(images_dir)
        except:
            os.mkdir(images_dir)
        
        print('starting {} images processing from {}'.format(day, root_dir+'/'+day+'_raw_images/'))
        for i in df['name']:
            for o in orientation:
                imgpath = os.getcwd()+'/raw_images/'+day+'_raw_images/'+i[:3]+'/'+o+'/'+i+'_'+o+'.jpg'
                if os.path.exists(imgpath):
                    count = count + 1
                    img = cv2.imread(imgpath)
                    if img.shape != (4000, 3000, 3):
                        if img.shape[0] > 4000:
                            print('cond2: ', imgpath)
                            # print(img.shape)
                            img = cv2.resize(img, (3000, 4000), interpolation = cv2.INTER_AREA)
                            print("after resizing: ",img.shape)
                        else:
                            # print('cond1 ', imgpath)
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            # print(img.shape)
                    cv2.imwrite(images_dir+'/'+i+'_'+o+'.jpg', img)
                else:
                    print("doesn't exists: ", imgpath)
        print('\n{} {}_images proccessed and saved at: '.format(count, day), images_dir)
    except OSError as err:
        print('cannot create directory:', err)

    try:
        try:
            shutil.rmtree(os.path.join(root_dir,day+'_pred_images'))
            shutil.rmtree(os.path.join(root_dir,day+'_jsons'))
            os.mkdir(os.path.join(root_dir,day+'_pred_images'))
            os.mkdir(os.path.join(root_dir,day+'_jsons'))
            print('created pred_images and json directory for {} images'.format(day))
        except:
            os.mkdir(os.path.join(root_dir,day+'_pred_images'))
            os.mkdir(os.path.join(root_dir,day+'_jsons'))
            print('created pred_images and json directory for {} images'.format(day))
    except:
        print('cannot create directory for {} images'.format(day))


    df1 = pd.DataFrame(columns=['name', 'angle', 'angle_size', 'angle_width', 'angle_height',
                                'angle_area', 'front', 'front_size', 'front_width',
                                'front_height', 'front_area', 'top', 'top_size',
                                'top_width', 'top_height', 'top_area'])


    print("\nstarting potato segmentation ... of {} {}_images".format(count,day))


    count = 0

    for img in df['name']:
        key = []
        values = []
        for o in orientation:
            path = images_dir+'/'+img+'_'+o+'.jpg'
            if os.path.exists(path):
                key = img+'_'+o
                image = cv2.imread(path)
                values.append(path)
                values.append(image.shape)
                pred = model.predict(path)
                pred.save(root_dir+'/'+day+'_pred_images/'+key+'.jpg')
                json_object = json.dumps(pred.json(), indent = 4)
                with open(root_dir+'/'+day+'_jsons/'+key+'.json', "w") as outfile:
                    outfile.write(json_object)
                coordinates = pred.json()
                p = []
                width = coordinates['predictions'][0]['width']
                values.append(width)
                height = coordinates['predictions'][0]['height']
                values.append(height)
                for i in coordinates['predictions'][0]['points']:
                    p.append([i['x'], i['y']])
                coordinates = [[y,x] for [x,y] in p]
                polygon = Polygon(coordinates)
                values.append(polygon.area)
                count = count + 1
            else:
                print("doesn't exists: ",  path)
        # print('segmentation done for {} {}_images'.format(count,day))
        values.insert(0, img)
        df1.loc[len(df1)] = values
        
    print('\n{} {}_images segmented and saved at: '.format(count,day), root_dir+'/'+day+'_pred_images')
    print('{} {}_json saved at: '.format(count,day), root_dir+'/'+day+'_jsons')

    merged_df = df.merge(df1, how = 'inner', on = ['name'])
    df_path = os.path.join(root_dir,day+'_segmentation_results.xlsx')
    merged_df.to_excel(df_path, index=False)

    print('{} segmentation results saved at: {}'.format(day, df_path))

    return '{} images processing and segmentation completed'.format(day)
    
if __name__ == "__main__":
    start = time.time()
    with open('api.json') as json_file:
        api_key, project_name = json.load(json_file).values()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name )
    model = project.version(1).model
    print('model loaded\n')
    import concurrent.futures
    days = ['day1', 'day2', 'day3']
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os. cpu_count() + 4)) as executor:
        futures = []
        for day in days:
            print('adding {} to executor'.format(day))
            futures.append(executor.submit(processAndsegmentImages, day=day))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    end = time.time()
    tlapsed = end-start
    print('total time taken: {:.2f} minutes'.format(tlapsed/60))