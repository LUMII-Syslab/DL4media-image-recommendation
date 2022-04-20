from easynmt import EasyNMT
easynmt_model = EasyNMT('opus-mt')
print(easynmt_model.translate(['labdien'], source_lang = 'lv', target_lang = 'en'))