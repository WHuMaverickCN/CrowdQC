import json
import sys
from shapely.geometry import shape
try:
    from osgeo import ogr,osr
except:
    sys.exit('ERROR: 未找到 GDAL/OGR modules')

def transform_coordinates(input_ds, target_epsg):
    if input_ds == None:
        return ogr.GetDriverByName("Memory").CreateDataSource("")
    # 获取输入DataSource中的第一个图层
    input_layer = input_ds.GetLayerByIndex(0)

    # 创建目标投影坐标系
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)  # EPSG代码 32648

    # 创建坐标转换对象
    transform = osr.CoordinateTransformation(input_layer.GetSpatialRef(), target_srs)

    # 创建输出的DataSource对象
    output_ds = ogr.GetDriverByName("Memory").CreateDataSource("")

    # 创建输出图层，与输入图层具有相同的几何类型和字段定义
    output_layer = output_ds.CreateLayer(input_layer.GetName(),
                                          geom_type=input_layer.GetGeomType(),
                                          srs=target_srs)

    output_layer.CreateFields(input_layer.schema)

    # 遍历输入图层的要素，进行坐标转换并复制属性
    for feature in input_layer:
        geometry = feature.GetGeometryRef()
        if geometry!=None:
            geometry.Transform(transform)

            new_feature = ogr.Feature(output_layer.GetLayerDefn())
            new_feature.SetGeometry(geometry)

            for field in feature.keys():
                _field = feature.GetField(field)
                if isinstance(feature.GetField(field),list):
                    new_feature.SetField(field, _field.__str__())
                else:
                    new_feature.SetField(field, _field)


            output_layer.CreateFeature(new_feature)

    return output_ds

def get_geojson_item_from_ogr_datasource(hd_item):
    if type(hd_item.data)==ogr.DataSource and hd_item.data.name=='':
        geojson_data = __ogr_datasource_to_geojson_dict(hd_item.data)
    else:
        geojson_data = json.loads(hd_item.data.name)
    # return geojson_data['features']['geometry']
    return geojson_data
    
def get_featurecollection_extent(_fc):
    '''
    读取featurecollection，循环内部每个geometry，计算最大extent
    '''
    _outlier = []
    _count = 0
    _alter_count = 0
    _data_type = -1
    if '.geojson' in _fc:
        with open(_fc,'r') as geojson_item:
            geojson_data = json.load(geojson_item)
        _data_type = 1
    elif type(_fc)==ogr.DataSource:
        source_data = _fc
        _data_type = 2
    else:
        geojson_data = json.loads(_fc.name)
        _data_type = 1    
    if _data_type==1:
        _keys_name_list = list(geojson_data.keys())
        if "type" in _keys_name_list and geojson_data['type'] == "FeatureCollection":
            for item_feature in geojson_data["features"]:
                # print(item_feature["geometry"])
                if item_feature["geometry"]!={} and item_feature["geometry"]!= None:
                    geojson_geometry = shape(item_feature["geometry"])
                    extent = geojson_geometry.bounds
                    if _count == 0:
                        _outlier = list(extent)
                    _count += 1
                        
                    if extent[0]<_outlier[0]:
                        _outlier[0] = extent[0]
                        _alter_count += 1
                    if extent[1]<_outlier[1]:
                        _outlier[1] = extent[1]
                        _alter_count += 1
                    if extent[2]>_outlier[2]:
                        _outlier[2] = extent[2]
                        _alter_count += 1
                    if extent[3]>_outlier[3]:
                        _outlier[3] = extent[3]
                        _alter_count += 1

            return _outlier
    elif _data_type==2:
        for _layer in source_data:
            for _feature in _layer:
                _geometry = _feature.GetGeometryRef()
                # _name = _feature.GetField("id")
                extent = _geometry.GetEnvelope()
                if _count == 0:
                    _outlier = list(extent)
                _count += 1
                        
                if extent[0]<_outlier[0]:
                    _outlier[0] = extent[0]
                    _alter_count += 1
                if extent[1]<_outlier[1]:
                    _outlier[1] = extent[1]
                    _alter_count += 1
                if extent[2]>_outlier[2]:
                    _outlier[2] = extent[2]
                    _alter_count += 1
                if extent[3]>_outlier[3]:
                    _outlier[3] = extent[3]
                    _alter_count += 1
        return _outlier
    
def __ogr_datasource_to_geojson_dict(datasource):
    feature_collection_dict = {
        "type": "FeatureCollection",  # 将几何体导出为 WKT 格式
        "features": []
    }
    layer_count = datasource.GetLayerCount()
    # 创建一个字典来存储结果
    result_dict = {}

    # 遍历每个图层
    for layer_index in range(layer_count):
        layer = datasource.GetLayerByIndex(layer_index)

        # 获取图层的名字
        layer_name = layer.GetName()

        # 创建一个列表来存储该图层的要素
        features_list = []

        # 遍历图层中的每个要素
        for feature in layer:
            feature_dict = json.loads(feature.ExportToJson())
            features_list.append(feature_dict)
        feature_collection_dict["features"] = features_list
        return feature_collection_dict
    