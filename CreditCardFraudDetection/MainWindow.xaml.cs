using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace CreditCardFraudDetection
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private string TRAIN_DATA_FILEPATH = "creditcard.csv";
        //private string MODEL_FILEPATH = "MLModel.zip";

        private MLContext mlContext = new MLContext(seed: 1);

        private void btnDetectFraud_Click(object sender, RoutedEventArgs e)
        {

            // 載入訓練資料
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            //建立訓練管線
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // 執行訓練
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // 執行Cross Validation
            Evaluate(mlContext, trainingDataView, trainingPipeline);

            //取出第一筆訓練資料當做測試資料
            ModelInput sampleData= mlContext.Data.CreateEnumerable<ModelInput>(trainingDataView, false).First();

            // 載入訓練妥的機器學習模型
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            //輸入測試資料進行測試
            ModelOutput predictionResult = predEngine.Predict(sampleData);

            //顯示測試結果
            Trace.WriteLine($"\n\nActual Class: {sampleData.Class} \nPredicted Class: {predictionResult.Prediction}\n\n");

             // Save model
            //SaveModel(mlContext, mlModel, MODEL_FILEPATH, trainingDataView.Schema);
      }

        IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // 準備欲分析的訓練資料欄位 
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" });

            // 選擇Microsoft研發的LightGBM(Light Gradient Boosting Machine)演算法, 提供良好的執行效率和記憶體使用量
            var trainer = mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options() { NumberOfIterations = 150, LearningRate = 0.2001066f, NumberOfLeaves = 7, MinimumExampleCountPerLeaf = 10, UseCategoricalSplit = true, HandleMissingValue = false, MinimumExampleCountPerGroup = 100, MaximumCategoricalSplitPointCount = 16, CategoricalSmoothing = 10, L2CategoricalRegularization = 5, Booster = new GradientBooster.Options() { L2Regularization = 1, L1Regularization = 0 }, LabelColumnName = "Class", FeatureColumnName = "Features" });
            
            //串連訓練管線
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }

        void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Trace.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            //執行Cross Validation
            var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Class");
            //顯示Cross Validation的結果
            PrintBinaryClassificationFoldsAverageMetrics(crossValidationResults);
        }

        ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Trace.WriteLine("=============== Training  model ===============");
            //使用訓練資料訓練機器學習模型
            ITransformer model = trainingPipeline.Fit(trainingDataView);
            Trace.WriteLine("=============== End of training process ===============");
            return model;
        }

        //void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        //{
        //    // Save/persist the trained model to a .ZIP file
        //    Trace.WriteLine($"=============== Saving the model  ===============");
        //    mlContext.Model.Save(mlModel, modelInputSchema, modelRelativePath);
        //    Trace.WriteLine("The model is saved to {0}", modelRelativePath);
        //}

        //void PrintBinaryClassificationMetrics(BinaryClassificationMetrics metrics)
        //{
        //    Trace.WriteLine($"************************************************************");
        //    Trace.WriteLine($"*       Metrics for binary classification model      ");
        //    Trace.WriteLine($"*-----------------------------------------------------------");
        //    Trace.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
        //    Trace.WriteLine($"*       Auc:      {metrics.AreaUnderRocCurve:P2}");
        //    Trace.WriteLine($"************************************************************");
        //}

        void PrintBinaryClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValResults)
        {
            //取得Cross Validation每一回合的訓練結果
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            //取得Cross Validation每一回合的Accuracy值
            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);

            //計算平均的Accuracy
            var AccuracyAverage = AccuracyValues.Average();

            //依據平均的Accuracy計算標準差
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);

            //計算信心指數
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);

            Trace.WriteLine($"*************************************************************************************************************");
            Trace.WriteLine($"*       Metrics for Binary Classification model      ");
            Trace.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Trace.WriteLine($"*       Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})");
            Trace.WriteLine($"*************************************************************************************************************");
        }

        double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }
    }
}