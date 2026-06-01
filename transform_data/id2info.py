import pandas as pd

# 读取两个 CSV 文件
info_df = pd.read_csv('1aZeroMean.csv')
event_df = pd.read_csv('conop_id_species_no_420v1_0507_new.csv')
event_df = event_df.rename(columns={'ID': 'Event'})

# 合并，仅按 Fossil name
merged_df = pd.merge(info_df, event_df, on='Fossil name', how='right')

# 重命名并处理列
merged_df = merged_df.rename(columns={'Event_y': 'Event'})
if 'Event_x' in merged_df.columns:
    merged_df = merged_df.drop(columns=['Event_x'])

# 将 Event 列移到第一列
cols = merged_df.columns.tolist()
cols.remove('Event')
merged_df = merged_df[['Event'] + cols]

# 保存结果
merged_df.to_csv('info_new.csv', index=False)

# import pandas as pd

# # 读取两个 CSV 文件
# info_df = pd.read_csv('1aZeroMean.csv')
# event_df = pd.read_csv('conop_id_species_no_420v1_0507_new.csv')
# event_df = event_df.rename(columns={'ID': 'Event'})

# merged_df = pd.merge(info_df, event_df, on='Fossil name', how='right')

# # 重命名并处理列
# merged_df = merged_df.rename(columns={'Event_y': 'Event'})

# # 删除 Event_x（info.csv 中的 Event）
# if 'Event_x' in merged_df.columns:
#     merged_df = merged_df.drop(columns=['Event_x'])

# # 保存结果
# merged_df.to_csv('info_new.csv', index=False)


# # 重命名以便合并
# event_df = event_df.rename(columns={'ID': 'Event'})

# # 进行 merge，保留 info 中所有匹配的行
# merged_df = pd.merge(info_df, event_df, on=['Fossil name'], how='inner')

# # 最终格式调整，改回 ID
# merged_df = merged_df.rename(columns={'Event': 'ID'})

# # 保存到 info_new.csv
# merged_df.to_csv('info_new.csv', index=False)