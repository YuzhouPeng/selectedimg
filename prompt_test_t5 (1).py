import torch,os
from PIL import Image
from lavis.models import load_model_and_preprocess
import pandas as pd

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image

    
image_list = []
image_name_list = []

for parent,_,files in os.walk("./bdd_100_test"):
    for file in files:
        filepath = os.path.join(parent,file)
        image_list.append(filepath)
        image_name_list.append(file)

print(image_list)

typenames_t5 = ["pretrain_flant5xl", "caption_coco_flant5xl", "pretrain_flant5xxl"]
typenames = ["pretrain_opt2.7b", "caption_coco_opt2.7b", "pretrain_opt6.7b", "caption_coco_opt6.7b"]
# loads BLIP-2 pre-trained model

opt_models = []
opt_processor = []

t5_models = []
t5_processor = []


prompt_questions = ["describe the weather in this image?","describe the brightness and contrast in this image?","describe the road condition in this image?","describe the nature environment in this image?",
                    "describe the landscape in this image?","is there any pedestrian in this image? the number of pedestrian?","is there any vehicle in this image? tell me the number of vehicle",
                    "is there any scooter in this image? tell me the number of scooter?","is there any tricycle/bicycle in this image? tell me the number of tricycle/bicycle?",
                    "is there any building in this image? tell me the number of building?","describe the semantic information in this image?"]

caption_prompt_output_sets = ["image caption","weather","brightness and contrast","road condition","nature environment","landscape","pedestrian","vehicle","scooter","bicycle/tricycle","building","semantic information"]

caption_output_sets_data = []
# for i in range(len(caption_prompt_output_sets)):
#    caption_output_sets_data.append([caption_prompt_output_sets[i]])

print(caption_output_sets_data)

imagesets = ["image_names"]

raw_image = Image.open(image_list[0]).convert("RGB")
print(raw_image)

for type_name in typenames_t5:

    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type=type_name, is_eval=True, device=device)
    t5_models.append(model)
    t5_processor.append(vis_processors)



# read image to text
# t5
for i in range(len(image_list)):

    raw_image = Image.open(image_list[i]).convert("RGB")
    # prepare the image
    print("current num {}".format(i))
    print(image_name_list[i])
    for k in range(len(t5_models)):
        modelname = "blip2_t5"+typenames[k]
        print(modelname)
        imagesets.append(image_name_list[i]+" "+modelname)
        model,vis_processors = t5_models[k],t5_processor[k]
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        temp = []
        for j in  range(len(caption_prompt_output_sets)):
            if j==0:
                temp.append(model.generate({"image": image}))
            else:
                temp.append(model.generate({"image": image, "prompt": "Question: "+prompt_questions[j-1]}))
        caption_output_sets_data.append(temp)
        # print(model.generate({"image": image}))
        # print(model.generate({"image": image, "prompt": "Question: describe the weather in this image?"}))

        # print(model.generate({"image": image, "prompt": "Question: describe the brightness and contrast in this image?"}))
        # print(model.generate({"image": image, "prompt": "Question: describe the road condition in this image?"}))
        # print(model.generate({"image": image, "prompt": "Question: describe the nature environment in this image?"}))
    
        # print(model.generate({"image": image, "prompt": "Question: describe the landscape in this image?"}))
        # print(model.generate({"image": image, "prompt": "Question: is there any pedestrian in this image? the number of pedestrian?"}))
        # print(model.generate({"image": image, "prompt": "Question: is there any vehicle in this image? tell me the number of vehicle?"}))

        # print(model.generate({"image": image, "prompt": "Question: is there any scooter in this image? tell me the number of scooter?"}))
        # print(model.generate({"image": image, "prompt": "Question: is there any tricycle/bicycle in this image? tell me the number of tricycle/bicycle?"}))

        # print(model.generate({"image": image, "prompt": "Question: is there any building in this image? tell me the number of building?"}))

# for type_name in typenames:

#     model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type=type_name, is_eval=True, device=device)
#     opt_models.append(model)
#     opt_processor.append(vis_processors)

# for i in range(len(image_list)):

#     raw_image = Image.open(image_list[i]).convert("RGB")    
    
#     for k in range(len(opt_models)):
#         modelname = "blip2_opt"+typenames[k]
#         print(modelname)
#         imagesets.append(image_list[i]+modelname)
#         model,vis_processors = opt_models[k],opt_processor[k]
#         image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

#         for j in  range(len(caption_output_sets_data)):
#             if j==0:
#                 caption_output_sets_data[j].append(model.generate({"image": image}))
#             else:
#                 caption_output_sets_data[j].append({"image": image, "prompt": "Question: "+prompt_questions[j-1]})


    #     model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type=type_name, is_eval=True, device=device)
    # # prepare the image
    #     image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    #     print(model.generate({"image": image}))
    #     print("weather: {}".format(model.generate({"image": image, "prompt": "Question: describe the weather in this image?"})))

    #     print(model.generate({"image": image, "prompt": "Question: describe the brightness and contrast in this image?"}))
    #     print(model.generate({"image": image, "prompt": "Question: describe the road condition in this image?"}))
    #     print(model.generate({"image": image, "prompt": "Question: describe the nature environment in this image?"}))
    
    #     print(model.generate({"image": image, "prompt": "Question: describe the landscape in this image?"}))
    #     print(model.generate({"image": image, "prompt": "Question: is there any pedestrian in this image? the number of pedestrian?"}))
    #     print(model.generate({"image": image, "prompt": "Question: is there any vehicle in this image?the number of vehicle?"}))

    #     print(model.generate({"image": image, "prompt": "Question: is there any scooter in this image? tell me the number of scooter?"}))
    #     print(model.generate({"image": image, "prompt": "Question: is there any tricycle/bicycle in this image? tell me the number of tricycle/bicycle?"}))

    #     print(model.generate({"image": image, "prompt": "Question: is there any building in this image? tell me the style of build?"}))

for caption_output_sets in caption_output_sets_data:
    print(caption_output_sets)

df = pd.DataFrame(caption_output_sets_data,columns=caption_prompt_output_sets)
col_names = df.columns.tolist()
col_names.insert(0,imagesets[0])
df = df.reindex(columns=col_names)
df[imagesets[0]] = imagesets[1:]

df.to_excel("result_t5.xlsx",index=False)

