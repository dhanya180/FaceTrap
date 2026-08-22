package com.facetrap

import kotlin.math.exp

/** Logistic calibration fitted on the held-out validation split (C=10, seed=402). */
object TargetConfidence {
    private const val SLOPE = 10.582016f
    private const val INTERCEPT = -4.672831f

    fun fromCosine(cosineSimilarity: Float): Float =
        (1.0 / (1.0 + exp(-(SLOPE * cosineSimilarity + INTERCEPT).toDouble()))).toFloat()
}
