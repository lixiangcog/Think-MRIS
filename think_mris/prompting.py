from dataclasses import dataclass


@dataclass(frozen=True)
class ThinkMRISPromptTemplate:
    system_prompt: str = (
        "You are an expert multimodal assistant for Medical Referring Image Segmentation. "
        "Localize the medically referred target precisely and keep reasoning efficient."
    )
    user_template: str = (
        "{image_token}"
        "You are solving a Medical Referring Image Segmentation task. "
        "Please localize \"{question}\" with one tight bounding box and two interior key points per target. "
        "Compare the difference between candidate regions and find the most closely matched target. "
        "Reason briefly for easy cases and elaborate only when the case is difficult or uncertain. "
        "Output the thinking process in <think> </think> and the final answer in <answer> </answer> tags. "
        "Use bbox_2d=[x1,y1,x2,y2] and points_2d=[[x,y],[x,y]] in JSON. "
        "i.e., <think> reasoning process here </think>"
        "<answer>{answer_format}</answer>"
    )
    answer_format: str = (
        "[{\"bbox_2d\": [10,100,200,210], "
        "\"points_2d\": [[30,110],[120,180]]}]"
    )

    def format_user_prompt(self, question: str, include_image_token: bool = True) -> str:
        return self.user_template.format(
            question=question.lower().strip("."),
            answer_format=self.answer_format,
            image_token="<image>\n" if include_image_token else "",
        )

