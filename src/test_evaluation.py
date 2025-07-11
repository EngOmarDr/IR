from evaluation_utils import precision_at_k

# تجربة بسيطة أمام المعيد
predicted = ["d1", "d2", "d3", "d4"]
relevant = ["d1", "d3", "d5"]
print("✅ P@3 =", precision_at_k(predicted, relevant, k=3))
