from dataclasses import dataclass


@dataclass(frozen=True)
class ThinkMRISPromptTemplate:
    system_prompt: str = (
        "You are an expert multimodal assistant for Medical Referring Image Segmentation. "
        "Localize the medically referred target precisely and keep reasoning efficient."
    )
    user_template: str = (
        "<image>\n"
        "You are solving a Medical Referring Image Segmentation task. "
        "Please find \"{question}\" with bboxs and points. "
        "Compare the difference between candidate regions and find the most closely matched target. "
        "Reason briefly for easy cases and elaborate only when the case is difficult or uncertain. "
        "Output the thinking process in <think> </think> and the final answer in <answer> </answer> tags. "
        "Output the bbox(es) and point(s) inside the interested region in JSON format. "
        "i.e., <think> reasoning process here </think>"
        "<answer>{answer_format}</answer>"
    )
    answer_format: str = (
        "[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, "
        "{\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
    )

    def format_user_prompt(self, question: str) -> str:
        return self.user_template.format(
            question=question.lower().strip("."),
            answer_format=self.answer_format,
        )

