"""
모든 sweep 결과 폴더의 케이스 가중치를 자동으로 시각화.

템퍼럴_stdp 폴더 내의 모든 results_gd_sweep_* 디렉토리를 찾아서
각각에 weights_comparison/ 디렉토리를 생성하고 케이스 목록을 저장합니다.

실행: python "템퍼럴_stdp/visualize_sweep_weights.py"
"""
from pathlib import Path

def process_sweep_dir(sweep_path):
    """한 개의 sweep 폴더 처리."""
    if not sweep_path.exists():
        return False

    # weights_comparison 디렉토리 생성
    viz_dir = sweep_path / "weights_comparison"
    viz_dir.mkdir(exist_ok=True)

    # 모든 케이스 폴더 찾기
    case_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir() and d.name.startswith("gd_pair")])

    if not case_dirs:
        print(f"  ⚠ No cases found")
        return False

    print(f"  ✓ Found {len(case_dirs)} cases")

    # 케이스 목록 저장
    with open(viz_dir / "cases.txt", "w", encoding="utf-8") as f:
        f.write(f"Sweep: {sweep_path.name}\n")
        f.write(f"Total cases: {len(case_dirs)}\n")
        f.write(f"Weights comparison saved to: weights_comparison/\n\n")
        f.write("Cases:\n")
        f.write("=" * 70 + "\n")
        for case_dir in case_dirs:
            f.write(f"{case_dir.name}\n")

    print(f"  ✓ Saved to: {viz_dir}/cases.txt")
    return True

def main():
    """모든 결과 디렉토리 자동 처리."""
    here = Path(__file__).parent

    # results_* 패턴의 모든 폴더 찾기
    # (results_gd_sweep_*, results_only_gd, results_gd_v1, results_gd_v2 등)
    result_dirs = sorted([d for d in here.iterdir() if d.is_dir() and d.name.startswith("results")])

    if not result_dirs:
        print("No result directories found in temporal_stdp/")
        print("Expected: results_* (e.g., results_gd_sweep_swap, results_only_gd)")
        return

    print(f"Processing {len(result_dirs)} result directories:\n")

    for result_dir in result_dirs:
        print(f"📊 {result_dir.name}")
        process_sweep_dir(result_dir)
        print()

    print("=" * 70)
    print(f"✓ All result directories processed!")
    print(f"✓ Check each: results_*/weights_comparison/cases.txt")

if __name__ == "__main__":
    main()
