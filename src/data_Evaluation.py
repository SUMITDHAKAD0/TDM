from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

class Evaluation_Data:
    def __init__(self, real, synthetic) -> None:
        self.real_data = real
        self.synthetic_data = synthetic

    def evaluation(self):   
        # creating metadata object for single table
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.real_data)

        quality_report = evaluate_quality(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data, metadata=metadata)

        # column_report = quality_report.get_details(property_name='Column Shapes')
        print('quality_report \n',quality_report)
        # print('column_report \n', column_report)

        def show_distrubtion(col):
            fig = get_column_plot(
                real_data = self.real_data,
                synthetic_data = self.synthetic_data,
                column_name=col,
                metadata=metadata
            )
            fig.write_image(f"./results/{col}.png")
                
            # fig.show()
        for i in self.real_data.columns:
            # print(i)
            show_distrubtion(i)
        return None
