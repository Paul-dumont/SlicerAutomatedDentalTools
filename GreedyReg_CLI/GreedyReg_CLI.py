#!/usr/bin/env python-real

import os
import re
import sys
import shutil
import tempfile
import argparse
import subprocess
import logging

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("GreedyReg_CLI requires the 'nibabel' Python package, which is not "
          "installed in Slicer's Python environment. Open the GreedyReg module "
          "and re-run registration (it will offer to install it), or install it "
          "manually with: PythonSlicer -m pip install nibabel",
          file=sys.stderr, flush=True)
    sys.exit(1)

# ===== Logging Configuration =====
logger = logging.getLogger("GreedyReg_CLI")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Matches a leading letter(s)+number(s) patient ID, ignoring everything after
ID_PATTERN = re.compile(r'^([A-Za-z]+\d+)', re.IGNORECASE)


def findNiftiFiles(folder):
    ids = {}
    if not folder or not os.path.isdir(folder):
        return ids
    for fname in os.listdir(folder):
        if fname.endswith('.nii.gz') or fname.endswith('.nii'):
            m = ID_PATTERN.match(fname)
            if m:
                ids[m.group(1).upper()] = os.path.join(folder, fname)
    return ids


def findMatFiles(folder):
    ids = {}
    if not folder or not os.path.isdir(folder):
        return ids
    for fname in os.listdir(folder):
        if fname.endswith('.mat'):
            m = ID_PATTERN.match(fname)
            if m:
                ids[m.group(1).upper()] = os.path.join(folder, fname)
    return ids


def findPairs(t1Folder, t2Folder, maskFolder, initFolder):
    t1s = findNiftiFiles(t1Folder)
    t2s = findNiftiFiles(t2Folder)
    masks = findNiftiFiles(maskFolder) if maskFolder else {}
    inits = findMatFiles(initFolder) if initFolder else {}

    pairs = []
    for patientId in sorted(set(t1s.keys()) & set(t2s.keys())):
        pairs.append((
            patientId,
            t1s[patientId],
            t2s[patientId],
            masks.get(patientId),
            inits.get(patientId),
        ))
    return pairs


def writeIdentityInit(initPath):
    """Write a Greedy-format .mat init file holding identity, nudging the
    zero translation slightly so Greedy doesn't treat it as identity."""
    matrix = np.eye(4)
    matrix[0, 3] = 0.001
    with open(initPath, 'w') as f:
        for row in matrix:
            f.write(' '.join(str(v) for v in row) + '\n')


def binarizeMaskFile(srcPath, destPath):
    maskImg = nib.load(srcPath)
    maskData = (maskImg.get_fdata() > 0).astype(np.float32)
    newMask = nib.Nifti1Image(maskData, maskImg.affine)
    newMask.header.set_data_dtype(np.float32)
    nib.save(newMask, destPath)


def buildRegistrationCommand(greedyBinary, fixedPath, movingPath, warpPath, initPath,
                              metric, transformType, maskPath=None):
    dof = "6" if transformType == "Rigid" else "12"
    if metric == "NMI":
        metricArgs = ["-m", "NMI"]
    elif metric == "NCC":
        metricArgs = ["-m", "NCC", "4x4x4"]
    else:
        metricArgs = ["-m", "SSD"]
    cmd = [greedyBinary]
    cmd.extend(["-d", "3", "-a"])
    cmd.extend(metricArgs)
    cmd.extend(["-i", fixedPath, movingPath])
    cmd.extend(["-o", warpPath])
    cmd.extend(["-n", "100x100x50x25"])
    cmd.extend(["-e", "0.5"])
    cmd.extend(["-search", "100", "10", "20"])
    cmd.extend(["-dof", dof])
    cmd.extend(["-ia", initPath])
    if maskPath:
        cmd += ["-gm", maskPath]
    return cmd


def runGreedyCase(greedyBinary, fixedPath, movingPath, outputPath, warpPath, initPath,
                   metric, transformType, maskPath, timeout=600):
    cmd = buildRegistrationCommand(
        greedyBinary, fixedPath, movingPath, warpPath, initPath, metric, transformType, maskPath)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Greedy affine registration failed")

    resampleCmd = [greedyBinary, "-d", "3",
                   "-rf", fixedPath,
                   "-rm", movingPath, outputPath,
                   "-r", warpPath]
    result2 = subprocess.run(resampleCmd, capture_output=True, text=True, timeout=timeout)
    if result2.returncode != 0:
        raise RuntimeError(result2.stderr.strip() or "Greedy resampling failed")


def main(args):
    if not os.path.isfile(args.greedyBinary):
        logger.error(f"Greedy binary not found: {args.greedyBinary}")
        sys.exit(1)

    os.makedirs(args.outputFolder, exist_ok=True)

    pairs = findPairs(args.t1Folder, args.t2Folder, args.maskFolder, args.initFolder)
    if not pairs:
        logger.error(f"No matching T1/T2 pairs found between {args.t1Folder} and {args.t2Folder}")
        sys.exit(1)

    total = len(pairs)
    logger.info(f"Found {total} pair(s): {', '.join(p[0] for p in pairs)}")

    for i, (patientId, fixedPath, movingPath, maskPath, initPath) in enumerate(pairs):
        progress = i / total
        print(f"<filter-progress>{progress:.2f}</filter-progress>", flush=True)
        print(f"<filter-comment>Registering {patientId} ({i + 1}/{total})...</filter-comment>", flush=True)
        logger.info(f"Processing {patientId} ({i + 1}/{total})")

        caseTmpDir = tempfile.mkdtemp(prefix=f"greedyreg_{patientId}_")
        try:
            outputPath = os.path.join(args.outputFolder, f"{patientId}_registered.nii.gz")
            warpPath = os.path.join(args.outputFolder, f"{patientId}_warp.mat")

            resolvedInitPath = initPath
            if not resolvedInitPath:
                resolvedInitPath = os.path.join(caseTmpDir, "init.mat")
                writeIdentityInit(resolvedInitPath)

            resolvedMaskPath = None
            if maskPath:
                resolvedMaskPath = os.path.join(caseTmpDir, "mask.nii.gz")
                binarizeMaskFile(maskPath, resolvedMaskPath)

            runGreedyCase(
                args.greedyBinary, fixedPath, movingPath, outputPath, warpPath,
                resolvedInitPath, args.metric, args.transformType, resolvedMaskPath)

            logger.info(f"{patientId} done -> {outputPath}")
        except Exception as e:
            logger.error(f"FAILED on {patientId}: {e}")
            sys.exit(1)
        finally:
            shutil.rmtree(caseTmpDir, ignore_errors=True)

    print("<filter-progress>1.00</filter-progress>", flush=True)
    print(f"<filter-comment>Batch complete! {total} case(s) registered.</filter-comment>", flush=True)
    logger.info(f"Batch complete! {total} case(s) registered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('t1Folder', type=str)
    parser.add_argument('t2Folder', type=str)
    parser.add_argument('maskFolder', type=str)
    parser.add_argument('initFolder', type=str)
    parser.add_argument('outputFolder', type=str)
    parser.add_argument('greedyBinary', type=str)
    parser.add_argument('metric', type=str)
    parser.add_argument('transformType', type=str)

    main(parser.parse_args())
