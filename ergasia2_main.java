/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ergasia2pkg;

import java.util.HashMap;
import java.util.Map;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.meta.HOMER;
import mulan.classifier.lazy.IBLR_ML;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.trees.J48;

/**
 *
 * @author nickfortress
 * @author longbow
 */
public class ergasia2_main {
    public static void ML_RUS_exec() throws InvalidDataFormatException, Exception {

        ML_RUS transformer_under = new ML_RUS(0.1f);

        MultiLabelInstances mli = new MultiLabelInstances("enron.arff", "enron.xml");
        MultiLabelInstances mliCloneUnder = transformer_under.transformInstances(mli);

        RAkEL rak = new RAkEL();
        HOMER hom = new HOMER();
        CalibratedLabelRanking cal = new CalibratedLabelRanking(new J48());
        IBLR_ML ibl = new IBLR_ML();
        Evaluator eval = new Evaluator();
        MultipleEvaluation evaluationsUnder_rak = eval.crossValidate(rak, mliCloneUnder, 10);
        MultipleEvaluation evaluationsUnder_hom = eval.crossValidate(hom, mliCloneUnder, 10);
        MultipleEvaluation evaluationsUnder_cal = eval.crossValidate(cal, mliCloneUnder, 10);
        MultipleEvaluation evaluationsUnder_ibl = eval.crossValidate(ibl, mliCloneUnder, 10);
        System.out.println("========== ML-Undersampling Evaluations (RAkEL) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_rak.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (HOMER) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_hom.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (CalibratedLabelRanking) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_cal.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (IBLR_ML) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_ibl.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Macro-averaged F-Measure"));
    }
    public static void ML_ROS_exec() throws InvalidDataFormatException, Exception {

        ML_ROS transformer_over = new ML_ROS(0.1f);

        MultiLabelInstances mli = new MultiLabelInstances("CAL500.arff", "CAL500.xml");
        MultiLabelInstances mliCloneOver = transformer_over.transformInstances(mli);

        RAkEL rak = new RAkEL();
        HOMER hom = new HOMER();
        CalibratedLabelRanking cal = new CalibratedLabelRanking(new J48());
        IBLR_ML ibl = new IBLR_ML();
        Evaluator eval = new Evaluator();
        MultipleEvaluation evaluationsUnder_rak = eval.crossValidate(rak, mliCloneOver, 10);
        MultipleEvaluation evaluationsUnder_hom = eval.crossValidate(hom, mliCloneOver, 10);
        MultipleEvaluation evaluationsUnder_cal = eval.crossValidate(cal, mliCloneOver, 10);
        MultipleEvaluation evaluationsUnder_ibl = eval.crossValidate(ibl, mliCloneOver, 10);
        System.out.println("========== ML-Undersampling Evaluations (RAkEL) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_rak.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (HOMER) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_hom.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (CalibratedLabelRanking) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_cal.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (IBLR_ML) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_ibl.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Macro-averaged F-Measure"));
    }

    public static void LP_ROS_exec() throws InvalidDataFormatException, Exception {
        LP_ROS transformer_over = new LP_ROS(0.1f);

        MultiLabelInstances mli = new MultiLabelInstances("CAL500.arff", "CAL500.xml");
        MultiLabelInstances mliCloneOver = transformer_over.transformInstances(mli);

        RAkEL rak = new RAkEL();
        HOMER hom = new HOMER();
        CalibratedLabelRanking cal = new CalibratedLabelRanking(new J48());
        IBLR_ML ibl = new IBLR_ML();
        Evaluator eval = new Evaluator();
        MultipleEvaluation evaluationsUnder_rak = eval.crossValidate(rak, mliCloneOver, 10);
        MultipleEvaluation evaluationsUnder_hom = eval.crossValidate(hom, mliCloneOver, 10);
        MultipleEvaluation evaluationsUnder_cal = eval.crossValidate(cal, mliCloneOver, 10);
        MultipleEvaluation evaluationsUnder_ibl = eval.crossValidate(ibl, mliCloneOver, 10);
        System.out.println("========== ML-Undersampling Evaluations (RAkEL) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_rak.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (HOMER) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_hom.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (CalibratedLabelRanking) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_cal.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (IBLR_ML) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_ibl.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Macro-averaged F-Measure"));
    }
    public static void LP_RUS_exec() throws InvalidDataFormatException, Exception {
        LP_RUS transformer_under = new LP_RUS(0.1f);

        MultiLabelInstances mli = new MultiLabelInstances("CAL500.arff", "CAL500.xml");
        MultiLabelInstances mliCloneUnder = transformer_under.transformInstances(mli);

        RAkEL rak = new RAkEL();
        HOMER hom = new HOMER();
        CalibratedLabelRanking cal = new CalibratedLabelRanking(new J48());
        IBLR_ML ibl = new IBLR_ML();
        Evaluator eval = new Evaluator();
        MultipleEvaluation evaluationsUnder_rak = eval.crossValidate(rak, mliCloneUnder, 10);
        MultipleEvaluation evaluationsUnder_hom = eval.crossValidate(hom, mliCloneUnder, 10);
        MultipleEvaluation evaluationsUnder_cal = eval.crossValidate(cal, mliCloneUnder, 10);
        MultipleEvaluation evaluationsUnder_ibl = eval.crossValidate(ibl, mliCloneUnder, 10);
        System.out.println("========== ML-Undersampling Evaluations (RAkEL) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_rak.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_rak.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (HOMER) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_hom.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_hom.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (CalibratedLabelRanking) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_cal.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_cal.getMean("Macro-averaged F-Measure"));
        System.out.println("========== ML-Undersampling Evaluations (IBLR_ML) =========");
        System.out.println("Example-Based Accuracy: " + evaluationsUnder_ibl.getMean("Example-Based Accuracy"));
        System.out.println("Micro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Micro-averaged F-Measure"));
        System.out.println("Macro-averaged F-Measure: " + evaluationsUnder_ibl.getMean("Macro-averaged F-Measure"));
    }

    public static void main(String args[]) throws Exception {
        ML_RUS_exec();
//        ML_ROS_exec();
//        LP_ROS_exec();
//        LP_RUS_exec();
    }

}
